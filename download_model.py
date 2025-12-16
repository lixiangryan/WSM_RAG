import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 設定模型 ID 與 本地儲存路徑 ---
EMBED_MODEL_ID = "BAAI/bge-m3"
EMBED_LOCAL_PATH = "./local_bge_m3"

RERANK_MODEL_ID = "BAAI/bge-reranker-base"
RERANK_LOCAL_PATH = "./local_bge_reranker"


def download():
    # 1. 下載 Embedding Model (Bi-Encoder)
    if os.path.exists(EMBED_LOCAL_PATH):
        print(f"✅ Embedding 模型已存在於: {EMBED_LOCAL_PATH}，跳過。")
    else:
        print(f"⬇️ 正在下載 Embedding 模型 {EMBED_MODEL_ID}...")
        try:
            model = SentenceTransformer(EMBED_MODEL_ID)
            model.save(EMBED_LOCAL_PATH)
            print(f"✅ Embedding 模型已儲存至: {EMBED_LOCAL_PATH}")
        except Exception as e:
            print(f"❌ Embedding 下載失敗: {e}")

    # 2. 下載 Reranker Model (Cross-Encoder)
    if os.path.exists(RERANK_LOCAL_PATH):
        print(f"✅ Reranker 模型已存在於: {RERANK_LOCAL_PATH}，跳過。")
    else:
        print(f"⬇️ 正在下載 Reranker 模型 {RERANK_MODEL_ID}...")
        try:
            # [重要] 這裡必須用 CrossEncoder 來下載，因為它是用來做重排序的
            model = CrossEncoder(RERANK_MODEL_ID)
            model.save(RERANK_LOCAL_PATH)
            print(f"✅ Reranker 模型已儲存至: {RERANK_LOCAL_PATH}")
        except Exception as e:
            print(f"❌ Reranker 下載失敗: {e}")


if __name__ == "__main__":
    download()
