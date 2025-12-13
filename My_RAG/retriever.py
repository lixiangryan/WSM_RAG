from typing import Any, Dict, List, Optional, Tuple
import os
import re
import numpy as np
import jieba
import re
from rank_bm25 import BM25Okapi
import ollama
import math
# ==========================================
# 1. Retriever Configuration
# ==========================================
class RAGConfig:
    SETTINGS = {
        "zh": {
            "vector_model": "qwen2.5:0.5b",       # 中文模型
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

    def search(self, query, top_k=50):
        # 處理 Query
        if self.tokenizer_type == "jieba":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.lower().split(" ")

        scores = self.bm25.get_scores(tokenized_query)
        
        # 格式化輸出: List of (index, score)
        results = []
        for idx, score in enumerate(scores):
            # 過濾掉分數極低或為 0 的結果
            if score > 1e-5: 
                results.append((idx, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class DenseRetriever:
    """
    模型 2: Vector Search 
    """
    def __init__(self, chunks, language):
        self.chunks = chunks
        self.language = language
        self.model = RAGConfig.get(language, "vector_model")
        self.embeddings = []
        
        # 程式剛開始跑，先花時間把所有文件轉換成向量存起來 (Indexing)
        self._build_index()

    def _get_embedding(self, text):
        #檢查Ollama
        if not ollama: return np.zeros(768)
        try:
            # 使用 Ollama API
            resp = ollama.embeddings(model=self.model, prompt=text)
            return np.array(resp["embedding"])
        except Exception as e:
            print(f"[Error] Embedding failed: {e}")
            return np.zeros(768)

    def _build_index(self):
        print(f"[{self.language}] Vector Indexing start...")
        for chunk in self.chunks:
            vec = self._get_embedding(chunk["page_content"])
            self.embeddings.append(vec)
        #將矩陣轉換成array
        self.embeddings = np.array(self.embeddings)

    def search(self, query, top_k=50):
        query_vec = self._get_embedding(query)

        #檢查是否轉換失敗或資料庫是空的
        if query_vec is None or len(self.embeddings) == 0:
            return []

        # Cosine Similarity 計算
        #query向量的長度（｜A｜）
        q_norm = np.linalg.norm(query_vec)
        #所有文章向量的長度（｜B｜）
        d_norms = np.linalg.norm(self.embeddings, axis=1)
        
        # 避免除以 0
        d_norms[d_norms == 0] = 1e-10
        if q_norm == 0: q_norm = 1e-10

        dot_products = np.dot(self.embeddings, query_vec)
        similarities = dot_products / (q_norm * d_norms)

        results = [(i, float(score)) for i, score in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# ==========================================
# 3. Re-ranking
# ==========================================

class RelevanceClassifier:
    """
    目前實作：基於規則 (Heuristic) 的分類器。
    進階實作：你可以訓練一個 Logistic Regression 或使用 Cross-Encoder 來取代這裡的邏輯。
    """
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
        """
        query_terms = set(re.findall(r"\w+", query.lower()))
        chunk_terms = set(re.findall(r"\w+", content_lower))
        if len(query_terms) > 0:
            overlap = len(query_terms & chunk_terms)
            overlap_ratio = overlap / len(query_terms) 
            boost += overlap_ratio * 0.05
        """

        final_score = original_score * (1 + boost)
        return final_score

# ==========================================
# 4. 主流程 (Main Pipeline)
#    對應作業：Fusion
# ==========================================

class EnsembleRetriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        self.weights = RAGConfig.get(language, "weights")
        
        # 初始化模型
        self.bm25_retriever = SparseRetriever(chunks, language)
        self.vector_retriever = DenseRetriever(chunks, language)
        self.classifier = RelevanceClassifier()

    def _normalize(self, results):
        """
        Normalization: 將分數映射到 [0, 1]
        BM25: 0 ~ inf -> 0 ~ 1
        Vector: -1 ~ 1 -> 0 ~ 1
        """
        if not results: return {}
        
        scores = [r[1] for r in results]
        min_s, max_s = min(scores), max(scores)
        
        norm_map = {}
        for idx, score in results:
            if max_s - min_s == 0:
                #避免除0
                norm_map[idx] = 1.0 if max_s > 0 else 0.0
            else:
                norm_map[idx] = (score - min_s) / (max_s - min_s)
        return norm_map

    def retrieve(self, query, top_k=10):
        # 1. 雙路召回 (Retrieval)
        # 為了融合效果，這裡取較多的候選集 (top_k * 3)
        candidates_k = top_k * 3
        bm25_res = self.bm25_retriever.search(query, top_k=candidates_k)
        vec_res = self.vector_retriever.search(query, top_k=candidates_k)

        # 2. 分數歸一化 (Normalization)
        bm25_norm = self._normalize(bm25_res)
        vec_norm = self._normalize(vec_res)

        # 3. 加權融合 (Weighted Sum Fusion)
        all_indices = set(bm25_norm.keys()) | set(vec_norm.keys())
        merged_results = []
        
        alpha = self.weights["vec"]
        beta = self.weights["bm25"]

        for idx in all_indices:
            s_bm25 = bm25_norm.get(idx, 0.0)
            s_vec = vec_norm.get(idx, 0.0)
            
            # hybrid+信心
            fusion_score = (beta * s_bm25) + (alpha * s_vec)
            if s_bm25 > 0 and s_vec > 0:
                fusion_score *= 1.1
            
            merged_results.append({
                "index": idx,
                "score": fusion_score,
                "chunk": self.chunks[idx]
            })

        # 4. Re-ranking (使用 Classifier / Heuristic)
        # 這是作業的 Advanced Task 部分
        for item in merged_results:
            new_score = self.classifier.predict_score(
                query, 
                item["chunk"]["page_content"], 
                item["score"]
            )
            item["score"] = new_score

        # 5. 最終排序
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        final_top_chunks = [item["chunk"] for item in merged_results[:top_k]]
        #assert final_top_chunks != [], "Final top chunks should not be empty."
        return final_top_chunks

def create_retriever(chunks, language):
    # assert chunks == [], "Chunks should not be empty."
    return EnsembleRetriever(chunks, language)