import json
import os
import re
import jieba.posseg as pseg
import math
from nltk.stem import PorterStemmer
from collections import defaultdict
from typing import List, Dict, Set, Any, Optional

class SimpleKnowledgeGraph:
    """
    A lightweight Knowledge Graph implementation using an inverted index structure.
    Nodes are Entities (Terms, Years) and Documents (Chunks).
    Edges represent the occurrence of an Entity in a Chunk.
    Supports saving/loading the index to/from a JSON file (Pre-computed KG).
    """
    def __init__(self, chunks: List[Dict[str, Any]], index_path: Optional[str] = None):
        self.chunks = chunks
        self.entity_map = defaultdict(set) # Entity -> Set[ChunkIndex]
        self.doc_freqs = defaultdict(int)  # Entity -> Document Frequency (DF)
        self.total_docs = len(chunks)
        self.stemmer = PorterStemmer()
        
        # Try to load pre-computed index if path is provided and exists
        index_loaded = False
        if index_path and os.path.exists(index_path):
            print(f"[KG] Loading pre-computed index from {index_path}...")
            if self.load(index_path):
                index_loaded = True

        if not index_loaded:
            print("[KG] Building graph from scratch (No valid index found)...")
            self._build_graph()

    def save(self, path: str):
        """Saves the entity_map and doc_freqs to a JSON file."""
        # Convert sets to lists for JSON serialization
        data = {
            "version": "2.0",
            "total_docs": self.total_docs,
            "entity_map": {k: list(v) for k, v in self.entity_map.items()},
            "doc_freqs": self.doc_freqs
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"[KG] Index (v2.0) saved to {path}")
        except Exception as e:
            print(f"[KG] Error saving index: {e}")

    def load(self, path: str) -> bool:
        """Loads the entity_map from a JSON file. Returns True if successful and valid."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check Version
            if data.get("version") != "2.0":
                print("[KG] Index version mismatch (expected 2.0). Rebuilding...")
                return False

            loaded_map = data["entity_map"]
            
            # Convert lists back to sets and Validate
            temp_map = defaultdict(set)
            max_chunk_idx = -1
            
            for k, v in loaded_map.items():
                chunk_indices = set(v)
                if chunk_indices:
                    max_idx = max(chunk_indices)
                    if max_idx > max_chunk_idx:
                        max_chunk_idx = max_idx
                temp_map[k] = chunk_indices
            
            # VALIDATION: Check if index matches current chunks
            if max_chunk_idx >= len(self.chunks):
                print(f"[KG] WARNING: Index mismatch! Index refers to chunk {max_chunk_idx}, but we only have {len(self.chunks)} chunks.")
                print("[KG] Discarding pre-computed index and falling back to runtime build.")
                return False
                
            self.entity_map = temp_map
            self.doc_freqs = defaultdict(int, data.get("doc_freqs", {}))
            self.total_docs = data.get("total_docs", len(self.chunks))
            
            print(f"[KG] Successfully loaded {len(self.entity_map)} entities. Index is valid.")
            return True
            
        except Exception as e:
            print(f"[KG] Error loading index: {e}, falling back to build from scratch.")
            return False

    def _is_contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        for ch in text:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    def _extract_entities(self, text: str, is_query: bool = False) -> Set[str]:
        """
        Extracts entities from text.
        
        Args:
            text: The text to extract from.
            is_query: If True, uses looser extraction rules.
        """
        entities = set()
        
        # 1. Extract Years (4-digit numbers) - Works for both languages
        # Fix: Use non-capturing group for prefix or capture full match
        years = re.findall(r"\b(?:19|20)\d{2}\b", text)
        for y in years:
            entities.add(f"Year:{y}")

        # 2. Chinese Entity Extraction
        if self._is_contains_chinese(text):
            # Use jieba POS tagging to extract specific entity types
            # nt: Organization, nr: Person, ns: Location, eng: English, n: Noun, vn: Verbal Noun
            words = pseg.cut(text)
            valid_pos = {'nt', 'nr', 'ns', 'eng', 'nz', 'n', 'vn'} 
            
            # Expanded Chinese Stopwords to filter generic nouns
            cn_stopwords = {
                "公司", "營收", "年報", "報告", "什麼", "多少", "為何", "如何",
                "金額", "單位", "新台幣", "部分", "情形", "年度", "權益", "影響", 
                "價值", "用途", "項目", "內容", "備註", "說明", "合計", "總計",
                "包含", "包括", "相關", "目前", "表示", "認為", "可能", "以及", 
                "除了", "之外", "因為", "所以", "如果", "但是", "可以", "能夠",
                "千元", "百分比", "附註", "詳信", "資訊", "資料", "表格", "我們"
            }
            
            for word, flag in words:
                # For Query, we accept 'x' (unknown) as well just in case
                if (flag in valid_pos or (is_query and flag in ['x'])) and len(word) > 1:
                     if word not in cn_stopwords:
                        entities.add(f"Term:{word.lower()}")
            
            # Also try to catch English terms in Chinese text using regex (often cleaner than jieba's 'eng')
            if is_query:
                 tokens = re.findall(r"\b[A-Za-z][a-zA-Z0-9&'\-\.]*\b", text)
            else:
                 tokens = re.findall(r"\b[A-Z][a-zA-Z0-9&'\-\.]*\b", text)
            
            for t in tokens:
                if len(t) > 2:
                    entities.add(f"Term:{t.lower()}")

            return entities

        # 3. English Entity Extraction (Original Logic)
        if is_query:
            # Looser regex for Query: Allow lowercase letters
            tokens = re.findall(r"\b[A-Za-z][a-zA-Z0-9&'\-\.]*\b", text)
        else:
            # Strict regex for Document Indexing: Capitalized words only
            tokens = re.findall(r"\b[A-Z][a-zA-Z0-9&'\-\.]*\b", text)
        
        # Expanded Stopwords list (Case-INsensitive checked below)
        stopwords = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "but", "with", "by", 
            "from", "as", "if", "while", "where", "when", "then", "it", "this", "that", "these", "those",
            "he", "she", "they", "we", "you", "i", "is", "are", "was", "were", "be", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must",
            "question", "answer", "context", "note", "table", "figure", "page",
            "how", "what", "which", "who", "whom", "whose", "why", "limit", "show", "tell", "me"
        }

        for t in tokens:
            t_lower = t.lower()
            # Filter distinct terms (length > 2) and skip stopwords
            if len(t) > 2 and t_lower not in stopwords and not t.isdigit():
                 # Stemming for English terms to improve recall (e.g. "investing" -> "invest")
                 stemmed_t = self.stemmer.stem(t_lower)
                 entities.add(f"Term:{stemmed_t}")

        return entities

    def _build_graph(self):
        """Constructs the Entity-Document graph."""
        for i, chunk in enumerate(self.chunks):
            text = chunk.get("page_content", "")
            # Indexing time: Strict Mode (is_query=False)
            entities = self._extract_entities(text, is_query=False)
            for ent in entities:
                self.entity_map[ent].add(i)
                self.doc_freqs[ent] += 1

    def search(self, query: str) -> Dict[int, float]:
        """
        Traverses the graph to find chunks related to entities in the query.
        Returns a dictionary of {chunk_index: score}.
        Score is currently based on the number of matched entities (Intersection).
        """
        # Query time: Loose Mode (is_query=True)
        query_entities = self._extract_entities(query, is_query=True)
        scores = defaultdict(float)
        
        for ent in query_entities:
            if ent in self.entity_map:
                # 1-Hop: Entity -> Chunks
                # If we had a deeper graph, we could do Entity -> Related Entity -> Chunks
                related_chunk_indices = self.entity_map[ent]
                
                # Weighting Refinement with IDF:
                # Base Weight: Term = 5.0, Year = 1.0
                base_weight = 5.0 
                if ent.startswith("Year:"):
                    base_weight = 1.0 
                
                # IDF Calculation: log(N / (df + 1)) + 1
                # Penalize very frequent terms, boost rare ones.
                df = self.doc_freqs.get(ent, 0)
                idf = math.log(1 + (self.total_docs / (df + 1)))
                
                final_weight = base_weight * idf

                for idx in related_chunk_indices:
                    scores[idx] += final_weight
                    
        return scores
