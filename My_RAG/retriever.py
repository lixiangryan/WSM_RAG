from typing import Any, Dict, List
import re
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
import requests
from ollama import Client
from utils import load_ollama_config

RERANK_API_URL = "http://ollama-gateway:11434/rerank"


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
            "bm25_tokenizer": "jieba",
            "weights": {"bm25": 0.4, "vec": 0.6},
        },
        "en": {
            "bm25_tokenizer": "regex",
            "weights": {"bm25": 0.5, "vec": 0.5},
        },
    }

    @classmethod
    def get(cls, lang: str, key: str):
        cfg = cls.SETTINGS.get(lang, cls.SETTINGS["en"])
        return cfg.get(key)


# ==========================================
# 2. 檢索模型 (Retrieval Models)
# ==========================================
class SparseRetriever:
    """BM25 sparse retriever"""

    def __init__(self, chunks: List[dict], language: str):
        self.chunks = chunks
        self.corpus = [chunk["page_content"] for chunk in chunks]
        self.language = language
        self.tokenizer_type = RAGConfig.get(language, "bm25_tokenizer")

        # 英文 regex tokenizer：英文單字 + 符號（符號通常可留可不留，你原本是保留）
        self.en_tokenizer_re = re.compile(r"\b\w+\b|[^\w\s]")

        if self.tokenizer_type == "jieba":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        elif self.tokenizer_type == "regex":
            self.tokenized_corpus = [
                [t.lower() for t in self.en_tokenizer_re.findall(doc)]
                for doc in self.corpus
            ]
        else:
            self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 50) -> Dict[int, float]:
        if not self.corpus:
            return {}

        if self.tokenizer_type == "jieba":
            tokenized_query = list(jieba.cut(query))
        elif self.tokenizer_type == "regex":
            tokenized_query = [t.lower() for t in self.en_tokenizer_re.findall(query)]
        else:
            tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)  # np.ndarray

        # 取 top_k 並只保留 > 0 的分數
        top_k = min(top_k, len(scores))
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        results: Dict[int, float] = {}
        for idx in top_k_indices:
            s = float(scores[idx])
            if s > 0:
                results[int(idx)] = s
        return results


class VectorRetriever:
    """
    Dense vector retriever using Ollama embeddings.

    注意：
    - 我們會做 L2 normalize，確保 dot product = cosine similarity
    - 相容兩種 Ollama Python client 介面：
        * client.embed(model=..., input=[...]) -> res["embeddings"] (batch)
        * client.embeddings(model=..., prompt="...") -> res["embedding"] (single)
    """

    def __init__(
        self,
        chunks: List[dict],
        language: str,
        client: Client,
        embedding_model: str = "qwen3-embedding:0.6b",
    ):
        self.chunks = chunks
        self.language = language
        self.embedding_model = embedding_model
        self.client = client
        self.corpus = [c["page_content"] for c in chunks]
        self.embeddings = self._index_documents()  # shape: (N, dim), normalized

    @staticmethod
    def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / (norm + eps)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """統一 embedding 呼叫，回 np.ndarray shape=(len(texts), dim)"""
        # 介面 1：embed(batch)
        if hasattr(self.client, "embed"):
            res = self.client.embed(model=self.embedding_model, input=texts)
            return np.array(res["embeddings"], dtype=np.float32)

        # 介面 2：embeddings(single) -> 逐筆
        if hasattr(self.client, "embeddings"):
            vecs = []
            for t in texts:
                r = self.client.embeddings(model=self.embedding_model, prompt=t)
                vecs.append(r["embedding"])
            return np.array(vecs, dtype=np.float32)

        raise RuntimeError("Ollama client has neither 'embed' nor 'embeddings' method.")

    def _index_documents(self) -> np.ndarray:
        print(f"[{self.language}] Vector Indexing start... ({len(self.corpus)} chunks)")
        if not self.corpus:
            return np.array([], dtype=np.float32)

        try:
            embeddings = self._embed_texts(self.corpus)
            embeddings = self._l2_normalize(embeddings)
            print(
                f"[SUCCESS] Vector model '{self.embedding_model}' indexed {len(embeddings)} chunks."
            )
            return embeddings
        except Exception as e:
            print(f"[Error] Embedding failed: {e}")
            return np.array([], dtype=np.float32)

    def search(self, query: str, top_k: int = 50) -> Dict[int, float]:
        if self.embeddings.size == 0:
            return {}

        try:
            q = self._embed_texts([query])  # shape (1, dim)
            q = self._l2_normalize(q)[0]  # shape (dim,)
        except Exception as e:
            print(f"[Error] Query Embedding failed: {e}")
            return {}

        # embeddings 都已 normalize，dot product = cosine similarity
        scores = self.embeddings @ q  # shape (N,)

        top_k = min(top_k, len(scores))
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        return {int(idx): float(scores[idx]) for idx in top_k_indices}


# ==========================================
# 3. Re-ranking
# ==========================================
class RemoteFlagReranker:
    """
    Remote reranker (batching + error handling)
    API payload:
      {"pairs":[{"text1":..., "text2":...}, ...]}
    Response:
      {"scores":[...]}
    """

    def __init__(self, api_url: str):
        self.api_url = api_url

    def compute_score(
        self, pairs: List[List[str]], max_length: int = 1024
    ) -> np.ndarray:
        MAX_PAIRS_PER_CALL = 32
        all_scores: List[float] = []

        for i in range(0, len(pairs), MAX_PAIRS_PER_CALL):
            batch_pairs = pairs[i : i + MAX_PAIRS_PER_CALL]
            payload = {
                "pairs": [
                    {"text1": a[:max_length], "text2": b[:max_length]}
                    for a, b in batch_pairs
                ]
            }

            try:
                resp = requests.post(self.api_url, json=payload, timeout=5)
                if resp.status_code != 200:
                    print(f"[Reranker API Error] ({resp.status_code}): {resp.text}")
                    all_scores.extend([0.0] * len(batch_pairs))
                    continue

                scores = resp.json().get("scores", [])
                if len(scores) != len(batch_pairs):
                    print(
                        "[Reranker API Error] scores length mismatch. Returning zeros for this batch."
                    )
                    all_scores.extend([0.0] * len(batch_pairs))
                    continue

                all_scores.extend([float(s) for s in scores])

            except requests.exceptions.RequestException as e:
                print(
                    f"[Reranker Connection Error] {e}. Returning zero scores for batch."
                )
                all_scores.extend([0.0] * len(batch_pairs))

        return np.array(all_scores, dtype=np.float32)


# ==========================================
# 4. 主流程 (Main Pipeline) - Fusion
# ==========================================
class EnsembleRetriever:
    def __init__(self, chunks: List[dict], language: str, client: Client):
        self.chunks = chunks
        self.language = language
        self.weights = RAGConfig.get(language, "weights") or {"bm25": 0.5, "vec": 0.5}

        self.sparse_retriever = SparseRetriever(chunks, language)
        self.vector_retriever = VectorRetriever(
            chunks, language, client, embedding_model="qwen3-embedding:0.6b"
        )
        self.classifier = RemoteFlagReranker(api_url=RERANK_API_URL)

    def _normalize(self, results: Dict[int, float]) -> Dict[int, float]:
        """Min-Max normalize to [0,1] for fusion."""
        if not results:
            return {}

        keys = list(results.keys())
        vals = np.array([results[k] for k in keys], dtype=np.float32)

        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if abs(vmax - vmin) < 1e-6:
            return {k: 1.0 for k in keys}

        norm = (vals - vmin) / (vmax - vmin)
        return {k: float(s) for k, s in zip(keys, norm)}

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        # 1) Sparse
        bm25_res = self.sparse_retriever.search(query, top_k=50)
        bm25_norm = self._normalize(bm25_res)

        # 2) Dense
        vec_res = self.vector_retriever.search(query, top_k=50)
        vec_norm = self._normalize(vec_res)

        # 3) Weighted fusion
        alpha = float(self.weights.get("vec", 0.5))
        beta = float(self.weights.get("bm25", 0.5))

        all_indices = set(bm25_norm.keys()) | set(vec_norm.keys())
        merged_results = []

        for idx in all_indices:
            s_bm25 = bm25_norm.get(idx, 0.0)
            s_vec = vec_norm.get(idx, 0.0)

            fusion_score = (beta * s_bm25) + (alpha * s_vec)
            if s_bm25 > 0 and s_vec > 0:
                fusion_score *= 1.1  # 你原本的信心加乘保留

            merged_results.append(
                {
                    "index": idx,
                    "score": float(fusion_score),
                    "chunk": self.chunks[idx],
                }
            )

        # 4) Rerank (cross-encoder)
        if merged_results:
            pairs_to_rerank = [
                [query, item["chunk"]["page_content"]] for item in merged_results
            ]
            rerank_scores = self.classifier.compute_score(pairs_to_rerank)

            # 安全：長度不一致就不覆蓋
            if len(rerank_scores) == len(merged_results):
                for i, item in enumerate(merged_results):
                    item["score"] = float(rerank_scores[i])

        # 5) Sort + TopK
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        return [item["chunk"] for item in merged_results[:top_k]]


def create_retriever(
    chunks: List[dict], language: str, client=None
) -> EnsembleRetriever:
    # 若外部沒傳 client，就自己用 config 建
    if client is None:
        cfg = load_ollama_config()
        client = Client(host=cfg["host"])
    return EnsembleRetriever(chunks, language, client)
