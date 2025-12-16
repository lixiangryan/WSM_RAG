# WSM RAG: Global Co-occurrence Graph (Phase 8)

**Baseline:** `origin/main` (Best Score Snapshot)  
**Date:** 2025-12-15 21:53 (UTC+8)  
**Branch:** `lixiang1213` (Integration)

This project integrates a **Global Entity Co-occurrence Graph** (Phase 8) into the high-performance WSM RAG baseline. It combines the robustness of Hybrid Retrieval (BM25 + Vector) with the precision of a pre-computed Knowledge Graph.

## 🏆 Architecture Overview

The system uses a **Weighted Sum Fusion** of three signals:

1.  **Sparse Retrieval (BM25):** captures keyword matches (Weight ~0.4).
2.  **Dense Retrieval (Vector):** captures semantic meaning (Weight ~0.6).
3.  **Knowledge Graph (Bonus Signal):** captures explicit entity relationships (Co-occurrence).

### Phase 8: Global Co-occurrence Graph (New!)
*   **V3.0 Indexing:** Pre-computed global co-occurrence map identifying statistically significant entity pairs across the entire corpus.
*   **Autonomous Expansion:** Instead of unstable runtime expansion, valid query terms (e.g., "TSMC") are expanded instantly using the global graph (e.g., -> "Wafer", "Revenue").
*   **Performance:** $O(1)$ lookup speed with zero runtime overhead.

## 📂 Key Files

*   `My_RAG/knowledge_graph.py`: **[NEW]** The V3.0 KG implementation.
*   `kg_index_en.json` / `kg_index_zh.json`: **[NEW]** Pre-computed KG indexes (~5303 entities).
*   `My_RAG/retriever.py`: **[MODIFIED]** Integration hook for KG Bonus Signal.
*   `My_RAG/main.py`: **[MODIFIED]** updated to pass index paths.
*   `scripts/build_kg_index.py`: Script to rebuild the graph from scratch.

## 🚀 How to Run

The workflow remains identical to the baseline. The system automatically detects and loads the `kg_index_*.json` files.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Inference
./run.sh
```

## 🛠️ Change Log

### 2025-12-15 (Phase 8 Integration)
*   **Baseline Reset**: Reset branch to `origin/main` to ensure a clean slate based on the best-scoring version.
*   **Feature Merge**: Cherry-picked Phase 8 KG files from `lixiang1202_optimize-rag-performance`.
*   **Integration**: Wired up `SimpleKnowledgeGraph` in `retriever.py` to provide a "Bonus Signal" (boosting relevance score if an entity match is found).
*   **Optimization**: Enabled "Index-Time Check" for query expansion to avoid runtime latencies.

### 2025-12-16 (Optimization Merge)
*   **Manual Optimization**: Cherry-picked critical optimizations from `origin/main`.
    *   **Metadata Injection**: Injected company/file names into chunks to improve retrieval context.
    *   **Performance Tuning**: Reduced Rerank Top-K from 50 to 15 (Critical Speedup).
    *   **Stability**: Reduced Vector Batch Size to 16.
