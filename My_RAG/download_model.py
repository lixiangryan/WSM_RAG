import os
import sys

# 嘗試匯入 sentence_transformers，如果沒裝會報錯提醒
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print(
        "❌ 錯誤: 尚未安裝 sentence-transformers。請先執行 'pip install -r requirements.txt'"
    )
    sys.exit(1)

# 嘗試匯入 ollama (用於下載 LLM)
try:
    import ollama
except ImportError:
    print("⚠️ 警告: 尚未安裝 ollama python 套件，將跳過 LLM 下載 (或改用系統指令)。")


def download_embedding_model():
    """
    下載 Sentence Transformer 模型 (用於 Chunking & Retrieval)
    這會將模型儲存在 Hugging Face 的預設 cache 目錄中。
    """
    # 這是你目前最強的選擇 (約 2.2GB)
    # 如果 5GB 空間爆了，請改用 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (約 470MB)
    MODEL_NAME = "BAAI/bge-m3"

    print(f"⬇️  正在下載 Embedding 模型: {MODEL_NAME} ...")
    try:
        # 這行程式碼執行後，模型檔案就會被永久存在硬碟裡
        model = SentenceTransformer(MODEL_NAME)
        print(f"✅ Embedding 模型 {MODEL_NAME} 下載完成！")
    except Exception as e:
        print(f"❌ Embedding 模型下載失敗: {e}")
        sys.exit(1)


def download_llm_model():
    """
    下載 Ollama 模型 (用於 Generation)
    """
    # 這是比賽規定的模型
    LLM_NAME = "granite4:3b"

    print(f"⬇️  正在透過 Ollama 下載 LLM: {LLM_NAME} ...")
    try:
        # 使用 Python SDK 下載 (需要先 pip install ollama)
        ollama.pull(LLM_NAME)
        print(f"✅ LLM 模型 {LLM_NAME} 下載完成！")
    except Exception as e:
        print(f"❌ LLM 下載失敗 (可能是 Ollama 服務沒開，或網路問題): {e}")
        # 這裡不一定要 exit，因為有時候是在 shell script 裡處理 ollama pull
        # 但為了保險，建議這裡要成功


if __name__ == "__main__":
    print("=== 開始下載模型 (準備進入斷網環境) ===")

    # 1. 下載 Embedding Model
    download_embedding_model()

    # 2. 下載 LLM (如果你是用 Ollama 跑生成)
    download_llm_model()

    print("=== 所有模型下載程序結束 ===")
