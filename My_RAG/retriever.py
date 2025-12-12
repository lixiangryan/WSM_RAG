from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import jieba
import re
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer
from ollama import Client
from generator import load_ollama_config
from knowledge_graph import SimpleKnowledgeGraph

import requests
import os
import torch

# 移除原本的 Ollama Client，因為 Rerank 不建議用生成式模型
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None

class RemoteFlagReranker:
    """
    Fake FlagReranker class: same interface as the official one (FlagEmbedding),
    but internally calls a remote API. Adapts to CrossEncoder interface via predict().
    """
    def __init__(self, api_url: str):
        self.api_url = api_url

    def compute_score(self, pairs, max_length=1024):
        # Strict Limit: Each API call supports at most 32 pairs.
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            payload = {"pairs": [{"text1": str(a)[:max_length], "text2": str(b)[:max_length]} for a, b in batch]}
            
            try:
                resp = requests.post(self.api_url, json=payload, timeout=30)
                if resp.status_code != 200:
                    print(f"Warning: Remote Reranker API failed ({resp.status_code}): {resp.text}")
                    # Return 0 scores for this batch on error to avoid crash
                    all_scores.extend([0.0] * len(batch))
                    continue
                    
                scores = resp.json().get("scores", [])
                all_scores.extend(scores)
            except Exception as e:
                print(f"Error calling Remote Reranker: {e}")
                all_scores.extend([0.0] * len(batch))

        return np.array(all_scores)

    def predict(self, pairs):
        # Alias for CrossEncoder compatibility
        return self.compute_score(pairs)
    
    @staticmethod
    def check_connectivity(api_url: str) -> bool:
        """Probe the API to see if it's reachable."""
        try:
            # Send a dummy pair to check connection
            payload = {"pairs": [{"text1": "test", "text2": "test"}]}
            resp = requests.post(api_url, json=payload, timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

class HybridRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "en",
        bm25_top_k: int = 50,      # BM25 取前 50
        embed_top_k: int = 50,     # 向量取前 50
        final_top_k: int = 5,      # 最終回傳前 5
        # rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", # 推薦的輕量級 Reranker
        rerank_model_name: str = "BAAI/bge-reranker-base", # 將預設模型換成支援中英雙語的 BAAI/bge-reranker-base
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_reranker: bool = True,
        index_path: Optional[str] = None
    ):
        self.language = language
        # 0. 初始化 Stemmer
        self.stemmer = PorterStemmer() if language != "zh" else None

        # 1. 過濾語言
        self.chunks = [
            c for c in chunks
            if not language
            or c.get("metadata", {}).get("language") == language
            or c.get("language") == language
        ]
        self.corpus = [chunk["page_content"] for chunk in self.chunks]
        
        self.bm25_top_k = bm25_top_k
        self.embed_top_k = embed_top_k
        self.final_top_k = final_top_k
        self.use_reranker = use_reranker
        
        self.ollama_client = None

        # 2. 初始化 BM25
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 3. 初始化 Embedding Model (Bi-Encoder)
        self.dense_encoder = None
        self.chunk_embeddings = None
        if SentenceTransformer:
            self.dense_encoder = SentenceTransformer(embedding_model_name)
            # 預計算所有 chunks 的向量 (實務上這步應該在存資料庫時就做好了，不要每次 init 做)
            # 這裡使用 encode(show_progress_bar=False) 且直接轉 numpy，比迴圈快
            self.chunk_embeddings = self.dense_encoder.encode(
                self.corpus, convert_to_numpy=True, normalize_embeddings=True
            )

        # 4. 初始化 Reranker (Hybrid: Local vs Remote)
        self.reranker = None
        if use_reranker:
            # Logic: 
            # 1. Check user preference (RERANKER_TYPE).
            # 2. If auto, check if Remote API is reachable (Submission Env often has weak GPU but working API).
            # 3. If Remote API is reachable -> Use Remote.
            # 4. If Remote API unreachable -> Fallback to Local GPU.
            
            reranker_type = os.getenv("RERANKER_TYPE", "auto").lower()
            remote_url = os.getenv("RERANKER_API_URL", "http://ollama-gateway:11434/rerank")
            
            use_remote = False
            
            if reranker_type == "remote":
                use_remote = True
            elif reranker_type == "local":
                use_remote = False
            else: # auto
                # Probe Remote API
                if RemoteFlagReranker.check_connectivity(remote_url):
                    print(f"[Reranker] Remote API detected at {remote_url}. Using Remote (Safer for Submission Env).")
                    use_remote = True
                else:
                    print(f"[Reranker] Remote API unreachable at {remote_url}. Falling back to Local.")
                    use_remote = False
            
            if use_remote:
                self.reranker = RemoteFlagReranker(remote_url)
            elif CrossEncoder:
                print(f"[Reranker] Using Local GPU: {rerank_model_name}")
                self.reranker = CrossEncoder(rerank_model_name)
            else:
                print("[Reranker] Warning: No GPU/CrossEncoder and no Remote API found. Reranking disabled or degraded.")
                # We could try Remote anyway or just fail gracefully?
                # Let's try Remote one last time in case probe failed flakily
                self.reranker = RemoteFlagReranker(remote_url)

        # 5. 初始化 Knowledge Graph (NEW)
        self.kg = SimpleKnowledgeGraph(chunks, index_path=index_path)

        try:
            config = load_ollama_config()
            ollama_host = config.get("host", "http://ollama-gateway:11434")
        except Exception:
            ollama_host = "http://ollama-gateway:11434"
            
        self.ollama_client = Client(host=ollama_host)

    def _tokenize(self, text: str):
        if self.language == "zh":
            return list(jieba.cut(text))
        # English: Split + Lowercase + Stemming
        # 這是 lixiang1202_optimize-rag-performance(1208)_part2 的優化：
        # 將單字還原回詞幹 (e.g., "paying", "pays" -> "pay")，增加檢索召回率
        tokens = text.split()
        return [self.stemmer.stem(t.lower()) for t in tokens]

    def _focus_terms(self, query):
        """Extract company-like tokens and years to boost exact matches."""
        # This logic is absorbed from the 'wang' branch to improve entity recall
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9&'\\-\\.]*|\\d{4}", query)
        lower_tokens = {t.lower() for t in tokens if len(t) > 2}
        years = {t for t in lower_tokens if t.isdigit()}
        names = lower_tokens - years
        return names, years

    def _rrf_fusion(self, bm25_indices: List[int], embed_indices: List[int], kg_indices: List[int], k: int = 60) -> List[int]:
        """
        Reciprocal Rank Fusion (RRF):
        不依賴絕對分數，而是依賴「排名」。解決了 BM25 分數和 Cosine 分數範圍不同的問題。
        """
        rrf_score = {}

        # 處理 BM25 排名
        for rank, idx in enumerate(bm25_indices):
            if idx not in rrf_score: rrf_score[idx] = 0
            rrf_score[idx] += 1 / (k + rank + 1)

        # 處理 Vector 排名
        for rank, idx in enumerate(embed_indices):
            if idx not in rrf_score: rrf_score[idx] = 0
            rrf_score[idx] += 1 / (k + rank + 1)

        # 處理 Knowledge Graph 排名
        for rank, idx in enumerate(kg_indices):
            if idx not in rrf_score: rrf_score[idx] = 0
            # KG 的權重可以調整，這裡假設它與其他兩者同等重要
            rrf_score[idx] += 1 / (k + rank + 1)

        # 根據 RRF 分數排序，由高到低
        sorted_indices = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, score in sorted_indices]

    def _generate_hyde_doc(self, query: str) -> str:
        """
        使用 LLM 生成一個假設性的答案 (Hypothetical Document)
        """
        prompt = (
            f"Please write a passage to answer the question\n"
            f"Question: {query}\n"
            f"Passage:"
        )
        try:
            # 這裡用一個比較快的小模型生成即可，例如 llama3:8b 或 gemma
            response = self.ollama_client.generate(model = "granite4:3b", prompt=prompt)
            return response.get("response", "").strip()
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return query  # 如果生成失敗，退回使用原始 query


    def _llm_cross_check(self, query: str, candidate_indices: List[int], initial_scores: List[float] = None) -> List[int]:
        """
        [1211_part2] LLM-based Reranker (Pointwise Scoring)
        Let the Generator Model (granite4:3b) verify the relevance of retrieved docs.
        """
        # Limit to Top-60 (RTX 5090 Ultra Mode)
        process_k = 60
        candidates = candidate_indices[:process_k]
        
        final_scored_results = []
        
        # Pointwise Scoring Prompt
        base_prompt = (
            "You are a relevance judge. Analyze the Document and the Query.\n"
            "Query: {query}\n"
            "Document: {snippet}\n"
            "Task: Rate the relevance from 0 to 10. Output ONLY the number.\n"
            "Score:"
        )

        for i, idx in enumerate(candidates):
            doc_content = self.corpus[idx]
            # Truncate content to avoid context overflow for simple judging
            snippet = doc_content[:500] 
            prompt = base_prompt.format(query=query, snippet=snippet)
            
            try:
                # Temperature=0.1 for stable scoring
                response = self.ollama_client.generate(
                    model="granite4:3b", 
                    prompt=prompt, 
                    options={"temperature": 0.1, "num_predict": 5}
                )
                
                # Parse Score
                score_str = response.get("response", "").strip()
                match = re.search(r"\d+", score_str)
                llm_score = int(match.group()) if match else 0
                
            except Exception:
                llm_score = 0
            
            # Hybrid Score: 70% Original (Ranker) + 30% LLM 
            # (Assuming initial_scores are available and normalized, strictly speaking Reranker output is logit, 
            # so we just use LLM score to re-sort for now or simple addition if we trust LLM highly)
            
            # Strategy: Just use LLM score to boost. 
            # If BGE says top, but LLM says 0, it drops.
            combined_score = llm_score 
            final_scored_results.append((idx, combined_score))

        # Sort by LLM score desc
        final_scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # If we processed fewer than total candidates, append the rest (unscored)
        ranked_indices = [x[0] for x in final_scored_results]
        if len(candidate_indices) > process_k:
            ranked_indices.extend(candidate_indices[process_k:])
            
        return ranked_indices[:self.final_top_k]

    def retrieve(self, query: str, top_k: int = None, use_hyde: bool = False) -> List[Dict[str, Any]]:
        final_k = top_k if top_k is not None else self.final_top_k
        
        # --- 1. 決定用於向量檢索的文字 ---
        search_text_for_vector = query
        if use_hyde and self.dense_encoder:
            # 如果開啟 HyDE，先生成假答案，用假答案去轉向量
            fake_doc = self._generate_hyde_doc(query)
            # print(f"HyDE Generated: {fake_doc[:50]}...") # debug用
            search_text_for_vector = fake_doc
        
        query_tokens = self._tokenize(query)
        
        # --- 階段 1: BM25 檢索 ---
        # get_scores 比較快，不要用 get_top_n
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # [Optimization from wang branch]
        # Light re-ranking: boost exact company/year matches to reduce entity confusion.
        names, years = self._focus_terms(query)
        if names or years:
            for i, chunk in enumerate(self.chunks):
                text_lower = chunk.get("page_content", "").lower()
                boost = 0.0
                if names:
                    boost += 0.3 * sum(1 for name in names if name in text_lower)
                if years:
                    boost += 0.15 * sum(1 for yr in years if yr in text_lower)
                if boost:
                    bm25_scores[i] += boost
        
        # 取得前 N 名的 index
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:self.bm25_top_k]

        # --- 階段 2: 向量檢索 (Vector Search) ---
        embed_top_indices = []
        if self.dense_encoder and self.chunk_embeddings is not None:
            query_embedding = self.dense_encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            
            # 使用矩陣運算一次計算所有相似度 (Dot Product 因為已 Normalize = Cosine Similarity)
            # 這是 Numpy 的廣播機制，比 for loop 快非常多
            similarities = np.dot(self.chunk_embeddings, query_embedding)
            embed_top_indices = np.argsort(similarities)[::-1][:self.embed_top_k]

        # --- 階段 3 (NEW): Knowledge Graph 檢索 ---
        kg_scores = self.kg.search(query)
        # Sort by score descending
        kg_sorted = sorted(kg_scores.items(), key=lambda x: x[1], reverse=True)
        # Take top K (e.g. same as BM25 top k for fusion pool)
        kg_top_indices = [idx for idx, score in kg_sorted[:self.bm25_top_k]]

        # --- 階段 4: 融合 (Hybrid Fusion) ---
        # 使用 RRF 合併三種結果 (BM25, Vector, KG)
        merged_indices = self._rrf_fusion(bm25_top_indices, embed_top_indices, kg_top_indices)
        
        # 這裡的候選集數量可以稍微多一點，例如取前 150 個給 Reranker，提升召回率 (Widen the Funnel)
        # Scale up for RTX 5090: Widen the funnel to 300
        candidate_indices = merged_indices[:300] 
        candidate_docs = [self.corpus[i] for i in candidate_indices]

        # --- 階段 5: 重排序 (Re-ranking) ---
        ranked_pool_indices = candidate_indices
        if self.reranker:
            # Cross-Encoder 接受 list of pairs: [(query, doc1), (query, doc2), ...]
            pairs = [[query, doc] for doc in candidate_docs]
            rerank_scores = self.reranker.predict(pairs)
            
            # 結合 index 和分數
            results_with_scores = list(zip(candidate_indices, rerank_scores))
            # 根據 rerank 分數重新排序
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 取前 100 名進入 LLM Final Check (5090 Power!)
            ranked_pool_indices = [idx for idx, score in results_with_scores[:100]]
        else:
            ranked_pool_indices = candidate_indices[:100]

        # --- 階段 6 (NEW): LLM Cross-Check ---
        # 這是 Part 2 的核心：讓 LLM 親自看一眼，確保真的相關
        final_indices = self._llm_cross_check(query, ranked_pool_indices)

        return [self.chunks[idx] for idx in final_indices]

# 使用範例
# 使用範例
def create_retriever(chunks, language="en", index_path=None):
    return HybridRetriever(chunks, language=language, index_path=index_path)
