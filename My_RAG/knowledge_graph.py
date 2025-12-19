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
        self.chunk_map = defaultdict(list) # ChunkIndex -> List[Entity] (Forward Index)
        self.doc_freqs = defaultdict(int)  # Entity -> Document Frequency (DF)
        self.co_occurrence_map = defaultdict(lambda: defaultdict(int)) # Entity -> {RelatedEntity -> Frequency}
        self.synonym_map = {} # Synonym -> Canonical Term
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
        """Saves the entity_map and co_occurrence_map to a JSON file."""
        # Convert defaultdicts to regular dicts for JSON
        co_occurrence_json = {k: dict(v) for k, v in self.co_occurrence_map.items()}
        
        data = {
            "version": "3.0",
            "total_docs": self.total_docs,
            "entity_map": {k: list(v) for k, v in self.entity_map.items()},
            "chunk_map": self.chunk_map,
            "doc_freqs": self.doc_freqs,
            "co_occurrence_map": co_occurrence_json
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            print(f"[KG] Index (v3.0) with Co-occurrence Graph saved to {path}")
        except Exception as e:
            print(f"[KG] Error saving index: {e}")

    def load(self, path: str) -> bool:
        """Loads the index from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check Version (We accept 3.0)
            if data.get("version") != "3.0":
                print(f"[KG] Index version mismatch (got {data.get('version')}, expected 3.0). Rebuilding...")
                return False

            loaded_map = data["entity_map"]
            
            # Convert lists back to sets
            temp_map = defaultdict(set)
            max_chunk_idx = -1
            
            for k, v in loaded_map.items():
                chunk_indices = set(v)
                if chunk_indices:
                    max_idx = max(chunk_indices)
                    if max_idx > max_chunk_idx:
                        max_chunk_idx = max_idx
                temp_map[k] = chunk_indices
            
            # Validation
            if max_chunk_idx >= len(self.chunks):
                print(f"[KG] WARNING: Index mismatch! Index refers to chunk {max_chunk_idx}, but we only have {len(self.chunks)} chunks.")
                return False
                
            self.entity_map = temp_map
            self.chunk_map = defaultdict(list, {int(k): v for k, v in data.get("chunk_map", {}).items()})
            self.doc_freqs = defaultdict(int, data.get("doc_freqs", {}))
            self.total_docs = data.get("total_docs", len(self.chunks))
            
            # Load Co-occurrence Map
            co_occurrence_data = data.get("co_occurrence_map", {})
            for k, v in co_occurrence_data.items():
                for neighbor, freq in v.items():
                    self.co_occurrence_map[k][neighbor] = freq

            print(f"[KG] Successfully loaded {len(self.entity_map)} entities and Co-occurrence Graph.")
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
        """Extracts entities from text."""
        entities = set()
        
        # 1. Extract Years (4-digit numbers)
        years = re.findall(r"\b(?:19|20)\d{2}\b", text)
        for y in years:
            entities.add(f"Year:{y}")

        # 2. Chinese Entity Extraction
        if self._is_contains_chinese(text):
            words = pseg.cut(text)
            valid_pos = {'nt', 'nr', 'ns', 'eng', 'nz', 'n', 'vn'} 
            cn_stopwords = {
                "公司", "營收", "年報", "報告", "什麼", "多少", "為何", "如何",
                "金額", "單位", "新台幣", "部分", "情形", "年度", "權益", "影響", 
                "價值", "用途", "項目", "內容", "備註", "說明", "合計", "總計",
                "包含", "包括", "相關", "目前", "表示", "認為", "可能", "以及", 
                "除了", "之外", "因為", "所以", "如果", "但是", "可以", "能夠",
                "千元", "百分比", "附註", "詳信", "資訊", "資料", "表格", "我們"
            }
            
            for word, flag in words:
                if (flag in valid_pos or (is_query and flag in ['x'])) and len(word) > 1:
                     if word not in cn_stopwords:
                        entities.add(f"Term:{word.lower()}")
            
            if is_query:
                 tokens = re.findall(r"\b[A-Za-z][a-zA-Z0-9&'\-\.]*\b", text)
            else:
                 tokens = re.findall(r"\b[A-Z][a-zA-Z0-9&'\-\.]*\b", text)
            
            for t in tokens:
                if len(t) > 2:
                    entities.add(f"Term:{t.lower()}")
            return entities

        # 3. English Entity Extraction
        if is_query:
            tokens = re.findall(r"\b[A-Za-z][a-zA-Z0-9&'\-\.]*\b", text)
        else:
            tokens = re.findall(r"\b[A-Z][a-zA-Z0-9&'\-\.]*\b", text)
        
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
            if len(t) > 2 and t_lower not in stopwords and not t.isdigit():
                 stemmed_t = self.stemmer.stem(t_lower)
                 entities.add(f"Term:{stemmed_t}")

        return entities

    def _build_graph(self):
        """Constructs the Entity-Document graph."""
        for i, chunk in enumerate(self.chunks):
            text = chunk.get("page_content", "")
            entities = self._extract_entities(text, is_query=False)
            self.chunk_map[i] = list(entities) # Store Forward Index
            for ent in entities:
                self.entity_map[ent].add(i)
                self.doc_freqs[ent] += 1
        
        # Build Co-occurrence Graph after basic graph is done
        self._build_co_occurrence_graph()

    def _build_co_occurrence_graph(self):
        """Builds the global entity co-occurrence map."""
        print("[KG] Building Global Co-occurrence Graph...")
        count = 0
        for chunk_idx, entity_list in self.chunk_map.items():
            # For every pair of entities in the chunk
            # O(N^2) where N is number of entities in a chunk (usually small < 20)
            sorted_ents = sorted(entity_list) # Sort to ensure consistent order if needed, or just iterate
            n = len(sorted_ents)
            if n < 2:
                continue
            
            for i in range(n):
                for j in range(i + 1, n):
                    ent_a = sorted_ents[i]
                    ent_b = sorted_ents[j]
                    
                    self.co_occurrence_map[ent_a][ent_b] += 1
                    self.co_occurrence_map[ent_b][ent_a] += 1
            count += 1
        print(f"[KG] Co-occurrence Graph built from {count} chunks.")

    def expand_query_globally(self, query_entities: Set[str], top_k: int = 3) -> Dict[str, float]:
        """
        Expands query entities using the GLOBAL co-occurrence graph.
        Returns a dict of {ExpandedEntity: Weight}.
        """
        candidates = defaultdict(int)
        
        for ent in query_entities:
            if ent in self.co_occurrence_map:
                neighbors = self.co_occurrence_map[ent]
                for neighbor, freq in neighbors.items():
                    if neighbor not in query_entities:
                        candidates[neighbor] += freq
        
        # Filter: Must co-occur at least 2 times globally (Noise filter)
        valid_candidates = {k: v for k, v in candidates.items() if v >= 2}
        
        if not valid_candidates:
            return {}

        # Sort by frequency
        sorted_candidates = sorted(valid_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Normalize weights (0.0 - 0.5)
        # Max freq might be huge, so just give a fixed discounted weight?
        # Or relative to max freq. Let's stick to fixed conservative weights for now.
        
        expansion_weights = {}
        for ent, freq in sorted_candidates:
            # Check if it's a "Year" -> Give slightly less weight to avoid year drift
            weight = 0.2
            if ent.startswith("Year:"):
                weight = 0.1
            expansion_weights[ent] = weight
            
        return expansion_weights

    def search(self, query: str, use_prf: bool = True) -> Dict[int, float]:
        """
        Traverses the graph to find chunks related to entities in the query.
        Uses Global Co-occurrence for Expansion.
        """
        # 1. Extract Entities from Query
        query_entities = self._extract_entities(query, is_query=True)
        
        # Synonym Expansion
        expanded_query_entities = set(query_entities)
        for ent in query_entities:
            term_body = ent.split(":", 1)[1] if ":" in ent else ent
            if term_body in self.synonym_map:
                canonical = self.synonym_map[term_body]
                expanded_query_entities.add(f"Term:{canonical}")
                stemmed_canonical = self.stemmer.stem(canonical.lower())
                expanded_query_entities.add(f"Term:{stemmed_canonical}")
        
        # Initial scoring with explicit query terms
        # Convert set to dict with weight 1.0
        entity_weights = {ent: 1.0 for ent in expanded_query_entities}
        
        # 2. Global Graph Expansion (Replaces Runtime PRF)
        if use_prf:
            expansion_dict = self.expand_query_globally(expanded_query_entities, top_k=3)
            # Merge expansion weights
            for ent, weight in expansion_dict.items():
                if ent not in entity_weights:
                    entity_weights[ent] = weight
                    
        # 3. Compute Final Scores
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
