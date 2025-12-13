from sentence_transformers import SentenceTransformer, util
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional

MAX_CHUNK_SIZE = 12

class Chunker:
    def __init__(self, model_name="BAAI/bge-m3", threshold=0.5):
        self.model = None
        self.threshold = threshold
        
        try:
            print(f"[Chunker] Attempting to load {model_name} from local cache...")
            self.model = SentenceTransformer(model_name, local_files_only=True)
            print(f"[Chunker] Successfully loaded {model_name}.")
        except Exception as e:
            print(f"[Chunker] ERROR: Local cache for {model_name} not found. ")
            print(f"[Chunker] STRICT OFFLINE MODE: Skipping online download to protect submission env.")
            print(f"[Chunker] Semantic Chunking disabled. Falling back to Basic Sliding Window.")
            self.model = None

    def spilt_txt_into_sentences(self, text, chunk_size=500, chunk_overlap=50):
        # 使用正則表達式進行句子切分 (保留標點)
        sentences = re.split(r"(?<=[。！？.!?\n])", text)
        return [s for s in sentences if s.strip()]

    def chunk_documents(self, docs, language):
        all_chunks = []
        
        # === Fallback Logic ===
        if self.model is None:
            # 使用 Basic Sliding Window 切分 (原始邏輯)
            print("[Chunker] Running Basic Sliding Window Chunking (Fallback)...")
            chunk_size = 1000
            chunk_overlap = 200
            for doc_index, doc in enumerate(docs):
                content = doc.get("content", "")
                if not content: continue
                
                text_len = len(content)
                start_index = 0
                while start_index < text_len:
                    end_index = min(start_index + chunk_size, text_len)
                    chunk_metadata = doc.copy()
                    chunk_metadata.pop("content", None)
                    
                    chunk = {
                        "page_content": content[start_index:end_index],
                        "metadata": chunk_metadata
                    }
                    all_chunks.append(chunk)
                    start_index += chunk_size - chunk_overlap
            return all_chunks
        
        # === Semantic Chunking Logic ===
        print("開始進行語意切分 (Semantic Chunking)...")
        # 這裡會下載或載入 BAAI/bge-m3，需確保有網路或已預下載
        
        for doc in tqdm(docs, desc="Processing Documents"):
            content = doc.get("content", "")
            if not content.strip():
                continue

            metadata = doc.copy()
            metadata.pop("content", None)

            sentences = [s for s in self.spilt_txt_into_sentences(content) if s.strip()]
            if not sentences:
                continue

            # 把句子轉成向量 (Semantic Embedding)
            embeddings = self.model.encode(sentences, convert_to_tensor=True)
            current_chunk = [sentences[0]]

            for i in range(len(sentences) - 1):
                # 計算相鄰句子的相似度
                score = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
                
                if score >= self.threshold:  # 如果相似度高於標準就合起來
                    current_chunk.append(sentences[i + 1])
                else:
                    # 相似度低，斷開，檢查當前 chunk 是否過短 (Noise filtering?)
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) > MAX_CHUNK_SIZE:
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
                if len(chunk_text) > MAX_CHUNK_SIZE:
                    all_chunks.append(
                        {
                            "page_content": chunk_text,
                            "metadata": metadata.copy(),
                        }
                    )

        return all_chunks

def chunk_documents(
    docs: List[Dict[str, Any]],
    language: Optional[str] = None,
    chunk_size: int = 1000,    # Unused in Semantic Chunking generally, but kept for interface
    chunk_overlap: int = 300,  # Unused
) -> List[Dict[str, Any]]:
    """
    使用基於 BAAI/bge-m3 的語意切分 (Semantic Chunking)。
    這會取代原本基於規則的切分。
    """
    # 初始化 Chunker (會載入模型)
    chunker = Chunker(model_name="BAAI/bge-m3", threshold=0.5)
    return chunker.chunk_documents(docs, language)
