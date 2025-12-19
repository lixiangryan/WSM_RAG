# My_RAG/chunker.py
from sentence_transformers import SentenceTransformer, util
import re
import os
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# 設定最小 Chunk 長度 (過濾雜訊用)
MIN_CHUNK_SIZE = 10
# 設定本地模型路徑 (必須與 download_model.py 一致)
LOCAL_MODEL_PATH = "./local_bge_m3"


class Chunker:
    def __init__(self, model_path, threshold=0.5):
        # 這裡傳入的一定要是本地路徑
        print(f"[Chunker] Loading model from local path: {model_path}")

        # device='cpu' 確保穩定，如果有 GPU 會自動用
        self.model = SentenceTransformer(model_path, device="cpu")
        self.threshold = threshold

    def spilt_txt_into_sentences(self, text):
        # 使用正則表達式切分句子，保留標點
        sentences = re.split(r"(?<=[。！？.!?\n])", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_documents(self, docs, language):
        all_chunks = []
        print(f"[{language}] 開始進行語意切分 (Semantic Chunking - Offline Mode)...")

        # 過濾空文檔
        valid_docs = [d for d in docs if d.get("content", "").strip()]

        for doc in tqdm(valid_docs, desc="Chunking", ncols=80, mininterval=1.0):
            content = doc.get("content", "")

            # Metadata Injection (公司名稱注入)
            company_name = doc.get("company_name", "") or doc.get(
                "fileName", ""
            ).replace(".pdf", "")
            if company_name and not content.startswith(f"【{company_name}】"):
                content = f"【{company_name}】 {content}"

            metadata = doc.copy()
            metadata.pop("content", None)

            sentences = self.spilt_txt_into_sentences(content)
            if not sentences:
                continue

            # 向量化 (這裡如果沒有本地模型，且 sentence-transformers 試圖連網，會失敗)
            try:
                embeddings = self.model.encode(
                    sentences, convert_to_tensor=True, show_progress_bar=False
                )
            except Exception as e:
                print(f"[Error] Encoding failed: {e}")
                continue

            current_chunk = [sentences[0]]

            for i in range(len(sentences) - 1):
                # 計算相鄰句子的相似度
                score = util.cos_sim(embeddings[i], embeddings[i + 1]).item()

                if score >= self.threshold:
                    current_chunk.append(sentences[i + 1])
                else:
                    chunk_text = "".join(current_chunk)
                    if len(chunk_text) > MIN_CHUNK_SIZE:
                        all_chunks.append(
                            {
                                "page_content": chunk_text,
                                "metadata": metadata.copy(),
                            }
                        )
                    current_chunk = [sentences[i + 1]]

            # 處理最後一個 chunk
            if current_chunk:
                chunk_text = "".join(current_chunk)
                if len(chunk_text) > MIN_CHUNK_SIZE:
                    all_chunks.append(
                        {
                            "page_content": chunk_text,
                            "metadata": metadata.copy(),
                        }
                    )

        return all_chunks


def chunk_documents(docs, language=None, chunk_size=None, chunk_overlap=None):
    """
    斷網模擬版入口函數
    """
    # 1. 嚴格檢查本地模型是否存在
    if not os.path.exists(LOCAL_MODEL_PATH):
        # 這是模擬斷網環境：找不到本地檔案就報錯，絕對不連網
        raise OSError(
            f"❌ [Offline Simulation Failed] 找不到本地模型資料夾: {LOCAL_MODEL_PATH}\n"
            f"請先執行 'python download_model.py' 下載模型！\n"
            f"在比賽中，這一步必須在 Build Phase 完成。"
        )

    # 2. 載入模型 (這時候已經保證路徑存在)
    chunker = Chunker(model_path=LOCAL_MODEL_PATH, threshold=0.5)

    return chunker.chunk_documents(docs, language)
