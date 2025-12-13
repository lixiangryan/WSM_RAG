
import os
import nltk
from sentence_transformers import SentenceTransformer

def download_models():
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    # 指定我們要預下載的模型
    models_to_download = [
        "sentence-transformers/all-MiniLM-L6-v2", # Retriever Embedding
        "BAAI/bge-m3"                               # Semantic Chunker
    ]
    
    for model_name in models_to_download:
        print(f"Downloading SentenceTransformer model: {model_name}...")
        SentenceTransformer(model_name)
    
    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
