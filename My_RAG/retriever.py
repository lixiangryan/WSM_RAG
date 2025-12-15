from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import os
import re
import numpy as np
import jieba
import math
from rank_bm25 import BM25Okapi



class RAGConfig:
    SETTINGS = {
        "zh": {
            "bm25_tokenizer": "jieba",
            "weights": {
                "bm25": 0.4,
                "vec": 0.6,
            },  # 如果你沒開 Vector，這個權重其實只影響排序分數的縮放
        },
        "en": {
            "bm25_tokenizer": "space",
            "weights": {"bm25": 0.5, "vec": 0.5},
        },
    }

    @classmethod
    def get(cls, lang, key):
        cfg = cls.SETTINGS.get(lang, cls.SETTINGS["en"])
        return cfg.get(key)


# ==========================================
# 2. 檢索模型 (Retrieval Models) - 修改重點
# ==========================================


class SparseRetriever:
    """
    BM25 - 優化版：支援分領域檢索 (Domain-Specific Retrieval)
    """

    def __init__(self, chunks, language):
        self.language = language
        self.tokenizer_type = RAGConfig.get(language, "bm25_tokenizer")

        # --- 修改 A: 建立分領域的索引 ---
        # 結構: self.domain_data = { "Finance": {"bm25": obj, "chunks": []}, ... }
        self.domain_data = {}

        # 1. 先將 chunks 依據 domain 分組
        grouped_chunks = defaultdict(list)
        for chunk in chunks:
            # 取得 domain，若無則歸類為 'general'
            domain = chunk.get("metadata", {}).get("domain", "general")
            grouped_chunks[domain].append(chunk)

        # 2. 為每個 domain 建立獨立的 BM25 索引
        for domain, d_chunks in grouped_chunks.items():
            corpus = [chunk["page_content"] for chunk in d_chunks]

            if self.tokenizer_type == "jieba":
                tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
            else:
                tokenized_corpus = [doc.lower().split(" ") for doc in corpus]

            self.domain_data[domain] = {
                "bm25": BM25Okapi(tokenized_corpus),
                "chunks": d_chunks,
            }
            # print(f"Built BM25 index for domain: {domain}, size: {len(d_chunks)}")

    def search(self, query, domain=None, top_k=50):
        # 如果有指定 domain 且該 domain 存在於索引中，就只搜該 domain
        if domain and domain in self.domain_data:
            target_domains = [domain]
        else:
            target_domains = list(self.domain_data.keys())

        # 處理 Query Tokenization (只做一次)
        if self.tokenizer_type == "jieba":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.lower().split(" ")

        all_results = []

        for target_domain in target_domains:
            bm25 = self.domain_data[target_domain]["bm25"]
            target_chunks = self.domain_data[target_domain]["chunks"]

            # 取得分數
            scores = bm25.get_scores(tokenized_query)

            # 收集結果
            for idx, score in enumerate(scores):
                if score > 1e-5:  # 過濾極低分
                    all_results.append((target_chunks[idx], score))

        # 3. 全域排序
        # 因為跨領域的分數可能不完全可比（BM25特性），但這是沒辦法中的辦法
        all_results.sort(key=lambda x: x[1], reverse=True)

        return all_results[:top_k]


class RelevanceClassifier:
    """
    保持原樣 (隊友的 Heuristic 邏輯)
    """

    def predict_score(self, query, chunk_content, original_score):
        if original_score == 0:
            return original_score

        boost = 0.0
        content_lower = chunk_content.lower()

        # Feature 1: 年份
        query_years = re.findall(r"\d{4}", query)
        if query_years:
            doc_years_all = re.findall(r"\d{4}", content_lower)
            num_years_in_doc = len(doc_years_all)
            match_count = sum(1 for y in query_years if y in content_lower)

            if match_count > 0:
                density_penalty = math.log(num_years_in_doc + 2)
                year_boost = (0.05 * match_count) / density_penalty
                boost += year_boost
            else:
                # 只有當年份不匹配時稍微加一點點通用分數?
                # 隊友邏輯保留：boost += 0.05
                boost += 0.05

        final_score = original_score * (1 + boost)
        return final_score


# ==========================================
# 4. 主流程 (Main Pipeline)
# ==========================================


class EnsembleRetriever:
    def __init__(self, chunks, language="en"):
        # 這裡不變
        self.chunks = chunks
        self.language = language
        self.weights = RAGConfig.get(language, "weights")

        # 初始化模型
        self.bm25_retriever = SparseRetriever(chunks, language)
        # self.vector_retriever = DenseRetriever(chunks, language) # 暫時關閉
        self.classifier = RelevanceClassifier()

    def _normalize(self, results):
        """
        results: List of (chunk, score)
        """
        if not results:
            return []  # 改為回傳 List 以配合後續邏輯

        scores = [r[1] for r in results]
        min_s, max_s = min(scores), max(scores)

        norm_results = []
        for chunk, score in results:
            if max_s - min_s == 0:
                norm_score = 1.0 if max_s > 0 else 0.0
            else:
                norm_score = (score - min_s) / (max_s - min_s)
            norm_results.append({"chunk": chunk, "score": norm_score})

        return norm_results

    # --- 修改 C: 增加 query_domain 參數 ---
    def retrieve(self, query, query_domain=None, top_k=10):
        # 1. 雙路召回 (傳入 domain)
        candidates_k = top_k * 3

        # 呼叫 BM25 並傳入 domain
        bm25_res = self.bm25_retriever.search(
            query, domain=query_domain, top_k=candidates_k
        )

        # 2. 分數歸一化
        # 注意：bm25_res 現在是 [(chunk, score), ...] 格式
        bm25_norm = self._normalize(bm25_res)

        # 3. 加權融合 (目前只有 BM25)
        merged_results = []

        # 因為只有 BM25，直接拿來用，但保留隊友的 fusion 結構以便未來擴充
        for item in bm25_norm:
            s_bm25 = item["score"]
            chunk = item["chunk"]

            fusion_score = s_bm25
            if s_bm25 > 0:
                fusion_score *= 1.1  # 隊友的信心加成

            merged_results.append({"chunk": chunk, "score": fusion_score})

        # 4. Re-ranking (Classifier)
        for item in merged_results:
            new_score = self.classifier.predict_score(
                query, item["chunk"]["page_content"], item["score"]
            )
            item["score"] = new_score

        # 5. 最終排序
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        final_top_chunks = [item["chunk"] for item in merged_results[:top_k]]

        return final_top_chunks


def create_retriever(chunks, language):
    return EnsembleRetriever(chunks, language)
