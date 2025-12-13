from sentence_transformers import SentenceTransformer
import os

try:
    print("Checking for BAAI/bge-m3...")
    model = SentenceTransformer("BAAI/bge-m3")
    print("SUCCESS: Model loaded from cache.")
    print(f"Model Path: {model.state_dict().keys()}") # Just to verify
except Exception as e:
    print(f"ERROR: {e}")
