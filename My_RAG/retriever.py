from typing import Any, Dict, List, Optional, Tuple
import os
import re
import numpy as np
import jieba
import re
from rank_bm25 import BM25Okapi
import ollama
import math
import requests

RERANK_API_URL = "http://ollama-gateway:11434/rerank"
# ==========================================
# 1. Retriever Configuration
# ==========================================
class RAGConfig:
    SETTINGS = {
        "zh": {
            "vector_model": "nomic-embed-text",       # 中文模型
            "bm25_tokenizer": "jieba",            
            "weights": {"bm25": 0.4, "vec": 0.6}, 
        },
        "en": {
            "vector_model": "nomic-embed-text",   # 英文模型
            "bm25_tokenizer": "space",
            "weights": {"bm25": 0.5, "vec": 0.5}, 
        }
    }

    #避免傳入未知語言（傳入例外語言視爲英文）
    @classmethod
    def get(cls, lang, key):
        cfg = cls.SETTINGS.get(lang, cls.SETTINGS["en"])
        return cfg.get(key)

# ==========================================
# 2. 檢索模型 (Retrieval Models)
# ==========================================

class SparseRetriever:
    """
    BM25
    """
    def __init__(self, chunks, language):
        #"page_content"抓出來
        self.corpus = [chunk["page_content"] for chunk in chunks]
        self.language = language
        self.tokenizer_type = RAGConfig.get(language, "bm25_tokenizer")
        
        # 建立索引
        if self.tokenizer_type == "jieba":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
            
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=50) -> Dict[int, float]:
        if self.tokenizer_type == "jieba":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.lower().split(" ")
            
        scores = self.bm25.get_scores(tokenized_query)
        
        results = {}
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        for idx in top_k_indices:
            # 只取分數大於 0 的結果
            if scores[idx] > 0:
                results[idx] = scores[idx]
        return results

class VectorRetriever:
    """
    向量檢索 (Vector Retrieval)
    使用 Ollama 進行 Embedding
    """
    def __init__(self, chunks, language, client):
        self.chunks = chunks
        self.language = language
        self.model_name = RAGConfig.get(language, "vector_model")
        self.client = client
        self.corpus = [chunk["page_content"] for chunk in chunks]
        self.embeddings = self._index_documents()
        
    def _index_documents(self):
        print(f"[{self.language}] Vector Indexing start...")
        try:
            # 這是 Ollama Embedding API 的呼叫方式
            res = self.client.embed(
                model=self.model_name,
                texts=self.corpus,
            )
            return np.array(res['embeddings'])
        except Exception as e:
            print(f"[Error] Embedding failed: {e}")
            return np.array([])
    
    def search(self, query: str, top_k=50) -> Dict[int, float]:
        if len(self.embeddings) == 0:
            return {}
        
        try:
            # 取得 Query 的 Embedding
            res = self.client.embed(
                model=self.model_name,
                texts=[query],
            )
            query_embedding = np.array(res['embeddings'][0])
        except Exception as e:
            print(f"[Error] Query Embedding failed: {e}")
            return {}
        
        # 計算餘弦相似度 (Cosine Similarity)
        # 簡單計算 dot product，因為向量已在 Ollama 服務器端正規化 (假設)
        # 也可以使用 sklearn.metrics.pairwise.cosine_similarity
        
        # 計算點積（與餘弦相似度成正比，因為都是正規化過的）
        scores = np.dot(self.embeddings, query_embedding)
        
        results = {}
        # 取得 Top-K 的索引
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        for idx in top_k_indices:
            # 相似度通常在 0 到 1 之間 (如果向量已正規化)
            results[idx] = scores[idx] 
            
        return results
# ==========================================
# 3. Re-ranking
# ==========================================
"""
class RelevanceClassifier:
    def predict_score(self, query, chunk_content, original_score):
        if original_score == 0:
            return original_score

        boost = 0.0
        content_lower = chunk_content.lower()
        
        # Feature 1: 年份
        query_years = re.findall(r"\d{4}", query)
        if query_years:
            #文章含有連續4個數字的數量
            doc_years_all = re.findall(r"\d{4}", content_lower)
            num_years_in_doc = len(doc_years_all)
            match_count = sum(1 for y in query_years if y in content_lower)

            if match_count > 0:
                # 懲罰係數：log(x + 2)
                density_penalty = math.log(num_years_in_doc + 2)
                year_boost = (0.05 * match_count) / density_penalty
                boost += year_boost
            
            else:
                boost += 0.05

        
        # Feature 2: 懲罰query
        query_terms = set(re.findall(r"\w+", query.lower()))
        chunk_terms = set(re.findall(r"\w+", content_lower))
        if len(query_terms) > 0:
            overlap = len(query_terms & chunk_terms)
            overlap_ratio = overlap / len(query_terms) 
            boost += overlap_ratio * 0.05
        

        final_score = original_score * (1 + boost)
        return final_score
"""

class RemoteFlagReranker:
    """
    Fake FlagReranker class: same interface as the official one (BAAI/bge-reranker-v2-m3), 
    but internally calls a remote API. 
    增加了批次處理 (Batching) 和錯誤處理。
    """
    def __init__(self, api_url: str):
        self.api_url = api_url

    def compute_score(self, pairs, max_length=1024):
        """
        pairs: list of [text1, text2]
        return: score of each pair in np.ndarray
        """
        MAX_PAIRS_PER_CALL = 32 # 助教提到的 API 限制
        all_scores = []
        
        # 實作批次處理 (Batching)
        for i in range(0, len(pairs), MAX_PAIRS_PER_CALL):
            batch_pairs = pairs[i:i + MAX_PAIRS_PER_CALL]
            
            # 轉換為 API 要求的 payload 格式
            payload = {"pairs": [{"text1": a, "text2": b} for a, b in batch_pairs]}

            try:
                # 設置 Timeout 以防止無限等待
                resp = requests.post(self.api_url, json=payload, timeout=5) 
                
                if resp.status_code != 200:
                    print(f"[Reranker API Error] Request failed ({resp.status_code}): {resp.text}")
                    # 如果 API 失敗，回傳零分，避免程式中斷
                    all_scores.extend(np.zeros(len(batch_pairs)).tolist())
                    continue
                
                scores = resp.json()["scores"]
                all_scores.extend(scores)

            except requests.exceptions.RequestException as e:
                 # 連線層級失敗，回傳零分
                 print(f"[Reranker Connection Error] Failed to connect to API: {e}. Returning zero scores for batch.")
                 all_scores.extend(np.zeros(len(batch_pairs)).tolist())
        
        return np.array(all_scores)
# ==========================================
# 4. 主流程 (Main Pipeline)
#    對應作業：Fusion
# ==========================================

class EnsembleRetriever:
    def __init__(self, chunks, language, client):
        self.chunks = chunks
        self.language = language
        self.weights = RAGConfig.get(language, "weights")
        
        # 初始化模型
        self.sparse_retriever = SparseRetriever(chunks, language)
        self.vector_retriever = VectorRetriever(chunks, language, client)
        self.classifier = RemoteFlagReranker(api_url=RERANK_API_URL)

    def _normalize(self, results: Dict[int, float]) -> Dict[int, float]:
        """將分數正規化到 [0, 1] 之間，用於融合"""
        if not results:
            return {}
        
        scores = np.array(list(results.values()))
        if np.max(scores) == 0:
            return {k: 0.0 for k in results.keys()}
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # 避免除以零
        if max_score - min_score < 1e-6:
            # 如果所有分數都一樣，直接設定為 0.5 (或 1.0)
            return {k: 1.0 for k in results.keys()}

        # Min-Max Normalization
        normalized_scores = (scores - min_score) / (max_score - min_score)
        
        return dict(zip(results.keys(), normalized_scores))

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        # 1. Sparse Retrieval
        bm25_res = self.sparse_retriever.search(query)
        bm25_norm = self._normalize(bm25_res)
        
        # 2. Dense Retrieval
        vec_res = self.vector_retriever.search(query)
        vec_norm = self._normalize(vec_res)

        # 3. 加權融合 (Weighted Sum Fusion)
        all_indices = set(bm25_norm.keys()) | set(vec_norm.keys())
        merged_results = []
        
        alpha = self.weights["vec"]
        beta = self.weights["bm25"]

        for idx in all_indices:
            s_bm25 = bm25_norm.get(idx, 0.0)
            s_vec = vec_norm.get(idx, 0.0)
            
            # hybrid+信心：雙方都有分數時給予加乘
            fusion_score = (beta * s_bm25) + (alpha * s_vec)
            if s_bm25 > 0 and s_vec > 0:
                fusion_score *= 1.1
            
            merged_results.append({
                "index": idx,
                "score": fusion_score,
                "chunk": self.chunks[idx]
            })

        # 4. Re-ranking (使用 RemoteFlagReranker 進行批次處理)
        if merged_results:
            # 建立 pairs 列表: [[query, chunk_content], [query, chunk_content], ...]
            pairs_to_rerank = []
            for item in merged_results:
                pairs_to_rerank.append([query, item["chunk"]["page_content"]])
            
            # 呼叫遠程 API 計算分數 (RemoteFlagReranker 內部已處理 batching)
            rerank_scores = self.classifier.compute_score(pairs_to_rerank) 

            # 將新的重排分數寫回 merged_results
            for i, item in enumerate(merged_results):
                # 直接採用 Re-ranker 的分數作為新的最終分數 (這是 Cross-Encoder 的標準做法)
                item["score"] = rerank_scores[i] 

        # 5. 最終排序
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        # 6. 返回 Top-K 結果
        final_chunks_to_return = []
        for item in merged_results[:top_k]:
            # item['chunk'] 才是包含 'page_content' 的原始 chunk 字典
            final_chunks_to_return.append(item['chunk']) 

        return final_chunks_to_return

def create_retriever(chunks, language, client) -> EnsembleRetriever:
    # assert chunks == [], "Chunks should not be empty."
    return EnsembleRetriever(chunks, language, client)