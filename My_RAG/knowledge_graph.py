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
        self.chunk_map = defaultdict(list) # ChunkIndex -> List[Entity] (Forward Index for PRF)
        self.doc_freqs = defaultdict(int)  # Entity -> Document Frequency (DF)
        self.synonym_map = {} # Synonym -> Canonical Term (e.g. "台積電" -> "TSMC")
        self.total_docs = len(chunks)
        self.stemmer = PorterStemmer()
        
        # Try to load pre-computed index if path is provided and exists
        index_loaded = False
        if index_path and os.path.exists(index_path):
            print(f"[KG] Loading pre-computed index from {index_path}...")
            if self.load(index_path):
                index_loaded = True
        
        # Try to load synonym map
        synonym_path = "synonym_map.json"
        if os.path.exists(synonym_path):
             try:
                 with open(synonym_path, 'r', encoding='utf-8') as f:
                     self.synonym_map = json.load(f)
                 print(f"[KG] Loaded {len(self.synonym_map)} synonyms from {synonym_path}")
             except Exception as e:
                 print(f"[KG] Warning: Failed to load synonym map: {e}")

        if not index_loaded:
            print("[KG] Building graph from scratch (No valid index found)...")
            self._build_graph()

    def save(self, path: str):
        """Saves the entity_map and doc_freqs to a JSON file."""
        # Convert sets to lists for JSON serialization
        data = {
            "version": "2.1",
            "total_docs": self.total_docs,
            "entity_map": {k: list(v) for k, v in self.entity_map.items()},
            "chunk_map": self.chunk_map,
            "doc_freqs": self.doc_freqs
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"[KG] Index (v2.1) saved to {path}")
        except Exception as e:
            print(f"[KG] Error saving index: {e}")

    def load(self, path: str) -> bool:
        """Loads the entity_map from a JSON file. Returns True if successful and valid."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check Version
            if data.get("version") != "2.1":
                print("[KG] Index version mismatch (expected 2.1). Rebuilding...")
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
            # Load Chunk Map (Convert keys to int because JSON keys are always strings)
            self.chunk_map = defaultdict(list, {int(k): v for k, v in data.get("chunk_map", {}).items()})
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
            # nt: Organization, nr: Person, ns: Location, eng: English, nz: Other Noun, n: Noun, vn: Verbal Noun
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
            self.chunk_map[i] = list(entities) # Store Forward Index
            for ent in entities:
                self.entity_map[ent].add(i)
                self.doc_freqs[ent] += 1

    def search(self, query: str, use_prf: bool = True) -> Dict[int, float]:
        """
        Traverses the graph to find chunks related to entities in the query.
        Supports Pseudo-Relevance Feedback (PRF) to expand query with related entities.
        """
        # 1. Initial Search
        query_entities = self._extract_entities(query, is_query=True)
        
        # Phase 6: Synonym Expansion
        # If a query entity is a known synonym, replace/add the canonical term
        expanded_query_entities = set(query_entities)
        for ent in query_entities:
            # Check Term:xxx
            term_body = ent.split(":", 1)[1] if ":" in ent else ent
            if term_body in self.synonym_map:
                canonical = self.synonym_map[term_body]
                # Add canonical term (assuming it's a Term)
                expanded_query_entities.add(f"Term:{canonical}")
                # Also try stemmed version of canonical
                stemmed_canonical = self.stemmer.stem(canonical.lower())
                expanded_query_entities.add(f"Term:{stemmed_canonical}")

        scores = self._compute_scores(expanded_query_entities)
        
        if not use_prf or not scores:
            return scores

        # 2. Pseudo-Relevance Feedback (PRF)
        # Get Top-3 Chunks from initial search
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_chunk_indices = [idx for idx, score in sorted_indices]
        
        # Mine frequent entities from these chunks
        candidate_entities = defaultdict(int)
        for idx in top_chunk_indices:
            # Use Forward Index to get entities in this chunk
            chunk_ents = self.chunk_map.get(idx, []) 
            for ent in chunk_ents:
                if ent not in query_entities: # Don't add what we already have
                    candidate_entities[ent] += 1
        
        # Select Top-3 expansion terms
        # Filter: Must appear in at least 2 of the top chunks if we have enough chunks, else just freq
        expansion_terms = sorted(candidate_entities.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 3. Re-score with Expansion
        # Add expansion terms with discounted weight (e.g., 0.5)
        expanded_entities = query_entities.copy()
        # We need to handle weighting in _compute_scores, so let's pass a weight map
        # But _compute_scores currently takes a set.
        # Let's refactor _compute_scores to take a Dict[Entity, WeightMultiplier]
        
        entity_weights = {ent: 1.0 for ent in query_entities}
        for ent, freq in expansion_terms:
            entity_weights[ent] = 0.5 # Discount factor for PRF terms
            
        final_scores = self._compute_scores(entity_weights)
        return final_scores

    def _compute_scores(self, entities_with_weights: Any) -> Dict[int, float]:
        """
        Helper to compute scores given a set of entities or a dict of {entity: weight_multiplier}.
        """
        scores = defaultdict(float)
        
        # Normalize input to Dict[Entity, Multiplier]
        if isinstance(entities_with_weights, set) or isinstance(entities_with_weights, list):
             target_entities = {ent: 1.0 for ent in entities_with_weights}
        else:
             target_entities = entities_with_weights

        for ent, multiplier in target_entities.items():
            if ent in self.entity_map:
                related_chunk_indices = self.entity_map[ent]
                
                # Base Weight
                base_weight = 5.0 
                if ent.startswith("Year:"):
                    base_weight = 1.0 
                
                # IDF Calculation
                df = self.doc_freqs.get(ent, 0)
                idf = math.log(1 + (self.total_docs / (df + 1)))
                
                final_weight = base_weight * idf * multiplier

                for idx in related_chunk_indices:
                    scores[idx] += final_weight
        return scores
