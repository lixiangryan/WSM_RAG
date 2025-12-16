from typing import Any, Dict, List, Optional, Tuple
import os
import re
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
import ollama
import math
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# 判斷是否在比賽環境
IS_COMPETITION_ENV = (
    "ollama-gateway" in OLLAMA_HOST or os.getenv("RAG_ENV") == "competition"
)

RERANK_API_URL = f"{OLLAMA_HOST}/rerank"

print(
    f"[Config] Host: {OLLAMA_HOST}, Mode: {'Competition (API)' if IS_COMPETITION_ENV else 'Local (CrossEncoder)'}"
)


# 1. Retriever Configuration
# 優先檢查本地模型，確保斷網能跑
if os.path.exists("./local_bge_m3"):
    EMBEDDING_MODEL = "./local_bge_m3"
    print("[Config] Using Local Embedding Model: ./local_bge_m3")
else:
    EMBEDDING_MODEL = "BAAI/bge-m3"
    print("[Config] Using HuggingFace Embedding Model: BAAI/bge-m3")


class RAGConfig:
    SETTINGS = {
        "zh": {
            "vector_model": EMBEDDING_MODEL,
            "bm25_tokenizer": "jieba",
            "weights": {"bm25": 0.4, "vec": 0.6},
        },
        "en": {
            "vector_model": EMBEDDING_MODEL,
            "bm25_tokenizer": "space",
            "weights": {"bm25": 0.5, "vec": 0.5},
        },
    }

    @classmethod
    def get(cls, lang, key):
        cfg = cls.SETTINGS.get(lang, cls.SETTINGS["en"])
        return cfg.get(key)


# ==========================================
# 2. 檢索模型 (Retrieval Models)
# ==========================================


class SparseRetriever:
    """BM25"""

    def __init__(self, chunks, language):
        self.corpus = [chunk["page_content"] for chunk in chunks]
        self.language = language
        self.tokenizer_type = RAGConfig.get(language, "bm25_tokenizer")

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
            if scores[idx] > 0:
                results[idx] = scores[idx]
        return results


class VectorRetriever:
    """向量檢索 (Vector Retrieval)"""

    def __init__(self, chunks, language, client):
        self.chunks = chunks
        self.language = language
        self.model_name = RAGConfig.get(language, "vector_model")
        self.client = client
        self.corpus = [chunk["page_content"] for chunk in chunks]
        self.embeddings = self._index_documents()

    def _index_documents(self):
        print(f"[{self.language}] Vector Indexing start with {self.model_name}...")
        try:
            # [Optimization] 使用 batch 處理以防記憶體爆炸
            batch_size = 16  # 稍微調小一點，比較安全
            all_embeddings = []
            for i in range(0, len(self.corpus), batch_size):
                batch = self.corpus[i : i + batch_size]
                # 這裡要小心，如果 ollama 版本太舊會沒有 embed 方法
                res = self.client.embed(model=self.model_name, input=batch)
                all_embeddings.extend(res["embeddings"])

            embeddings = np.array(all_embeddings)
            print(f"[SUCCESS] Vector model indexed {len(embeddings)} chunks.")
            return embeddings

        except AttributeError:
            print(
                f"[Fatal Error] 你的 Ollama 套件版本太舊，不支援 embed！請執行 pip install --upgrade ollama"
            )
            return np.array([])
        except Exception as e:
            print(f"[Error] Embedding failed: {e}")
            return np.array([])

    def search(self, query: str, top_k=50) -> Dict[int, float]:
        if len(self.embeddings) == 0:
            return {}
        try:
            res = self.client.embed(model=self.model_name, input=[query])
            query_embedding = np.array(res["embeddings"][0])
        except Exception as e:
            print(f"[Error] Query Embedding failed: {e}")
            return {}

        scores = np.dot(self.embeddings, query_embedding)
        results = {}
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        for idx in top_k_indices:
            results[idx] = scores[idx]
        return results


# ==========================================
# 3. Hybrid Reranker (Auto-Switch)
# ==========================================


class HybridReranker:

    def __init__(self, api_url: str):
        self.api_url = api_url
        self.is_api_mode = IS_COMPETITION_ENV
        self.local_model = None

        if not self.is_api_mode:
            print("[Reranker] Detected Local Mode. Loading local CrossEncoder...")
            try:
                from sentence_transformers import CrossEncoder

                # [Critical] 這裡如果斷網且沒下載模型，會直接掛掉。
                # 建議在 download_model.py 裡也要下載 'BAAI/bge-reranker-base'
                if os.path.exists("./local_bge_reranker"):
                    print("[Reranker] Loading from local storage...")
                    rerank_model_path = "./local_bge_reranker"
                else:
                    print("[Reranker] Local model not found, trying HuggingFace...")
                    rerank_model_path = "BAAI/bge-reranker-base"

                self.local_model = CrossEncoder(rerank_model_path, device="cpu")
            except Exception as e:
                print(
                    f"[Warning] Failed to load local Reranker: {e}. Reranking step will be SKIPPED."
                )
                self.local_model = None

    def compute_score(self, pairs):
        # 1. 本地模式 (Local Mode)
        if not self.is_api_mode:
            if self.local_model:
                try:
                    # [Optimization] 加入 batch_size 和隱藏進度條以加速
                    scores = self.local_model.predict(
                        pairs,
                        batch_size=4,
                        show_progress_bar=False,
                        num_workers=0,  # 避免多執行緒在某些環境下的 overhead
                    )
                    return np.array(scores)
                except Exception as e:
                    print(f"[Local Rerank Error] {e}")
                    return np.zeros(len(pairs))
            else:
                return np.zeros(len(pairs))  # Skip rerank if model not loaded

        # 2. 比賽 API 模式 (Competition Mode)
        MAX_PAIRS_PER_CALL = 32
        all_scores = []
        for i in range(0, len(pairs), MAX_PAIRS_PER_CALL):
            batch_pairs = pairs[i : i + MAX_PAIRS_PER_CALL]
            payload = {"pairs": [{"text1": a, "text2": b} for a, b in batch_pairs]}
            try:
                resp = requests.post(
                    self.api_url, json=payload, timeout=5  # 縮短 timeout 避免卡太久
                )
                if resp.status_code == 200:
                    scores = resp.json().get("scores", [])
                    all_scores.extend(scores)
                else:
                    print(f"[API Error] {resp.status_code}")
                    all_scores.extend(np.zeros(len(batch_pairs)).tolist())
            except Exception as e:
                print(f"[Connection Error] {e}")
                all_scores.extend(np.zeros(len(batch_pairs)).tolist())

        return np.array(all_scores)


# ==========================================
# 4. 主流程 (Main Pipeline)
# ==========================================


class EnsembleRetriever:
    def __init__(self, chunks, language, client):
        self.chunks = chunks
        self.language = language
        self.weights = RAGConfig.get(language, "weights")

        self.sparse_retriever = SparseRetriever(chunks, language)
        self.vector_retriever = VectorRetriever(chunks, language, client)
        self.reranker = HybridReranker(api_url=RERANK_API_URL)

    def _normalize(self, results: Dict[int, float]) -> Dict[int, float]:
        if not results:
            return {}
        scores = np.array(list(results.values()))
        if np.max(scores) == np.min(scores):
            return {k: 1.0 for k in results.keys()}
        return dict(
            zip(
                results.keys(),
                (scores - np.min(scores)) / (np.max(scores) - np.min(scores)),
            )
        )

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        # 1. 初步檢索
        bm25_res = self.sparse_retriever.search(query)
        vec_res = self.vector_retriever.search(query)

        bm25_norm = self._normalize(bm25_res)
        vec_norm = self._normalize(vec_res)

        # 2. 融合分數
        all_indices = set(bm25_norm.keys()) | set(vec_norm.keys())
        merged_results = []
        alpha, beta = self.weights["vec"], self.weights["bm25"]

        for idx in all_indices:
            s_bm25 = bm25_norm.get(idx, 0.0)
            s_vec = vec_norm.get(idx, 0.0)
            fusion_score = (beta * s_bm25) + (alpha * s_vec)
            if s_bm25 > 0 and s_vec > 0:
                fusion_score *= 1.1  # Hybrid Bonus

            merged_results.append(
                {"index": idx, "score": fusion_score, "chunk": self.chunks[idx]}
            )

        # 3. 重排序 (Rerank)
        merged_results.sort(key=lambda x: x["score"], reverse=True)

        # [Critical Optimization] 只對前 15 名進行 Rerank！
        # 之前是 50，這在 CPU 上會導致超時 (50s/it -> 10s/it)
        RERANK_TOP_K = 15
        rerank_candidates = merged_results[:RERANK_TOP_K]

        if rerank_candidates:
            pairs = [
                [query, item["chunk"]["page_content"]] for item in rerank_candidates
            ]
            new_scores = self.reranker.compute_score(pairs)

            for i, item in enumerate(rerank_candidates):
                if np.sum(np.abs(new_scores)) > 0:
                    item["score"] = new_scores[i]

        # 4. 再次排序並回傳 Top-K
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        return [item["chunk"] for item in merged_results[:top_k]]


def create_retriever(chunks, language, client) -> EnsembleRetriever:
    return EnsembleRetriever(chunks, language, client)
