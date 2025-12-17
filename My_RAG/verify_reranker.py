
import sys
try:
    from sentence_transformers import CrossEncoder
    print("Loading Reranker Model...")
    model = CrossEncoder('BAAI/bge-reranker-v2-m3')
    print("SUCCESS: Reranker Model Loaded.")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
