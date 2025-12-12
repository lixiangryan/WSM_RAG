
import os
import nltk
from sentence_transformers import SentenceTransformer

def download_models():
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    # 指定我們要預下載的模型
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Downloading SentenceTransformer model: {model_name}...")
    
    # 這會將模型下載到預設的 cache 目錄 (/root/.cache/huggingface/hub)
    # 之後在 Runtime 載入時，只要 cache_folder 沒變，它就會直接用
    model = SentenceTransformer(model_name)
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_models()
