from sentence_transformers import SentenceTransformer, util
import re
from tqdm import tqdm

MAX_CHUNK_SIZE = 12


class Chunker:
    def __init__(self, model_name="BAAI/bge-m3", threshold=0.5):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold  # 低於 threshold 的句子會被合併

    def spilt_txt_into_sentences(self, text, chunk_size=500, chunk_overlap=50):
        sentences = re.split(r"(?<=[。！？.!?\n])", text)
        # /?<= 代表切完後要把標點符號留在前一個句子
        return [s for s in sentences if s.strip()]

    def chunk_documents(self, docs, language):
        all_chunks = []
        print("開始進行語意切分 (Semantic Chunking)...")
        for doc in tqdm(docs, desc="Processing Documents"):
            content = doc.get("content", "")
            if not content.strip():
                continue

            metadata = doc.copy()
            metadata.pop("content", None)

            sentences = [s for s in self.spilt_txt_into_sentences(content) if s.strip()]
            # 先把抓下來的文章切成句子
            if not sentences:
                continue

            # 把句子轉成向量
            embeddings = self.model.encode(sentences, convert_to_tensor=True)
            current_chunk = [sentences[0]]

            for i in range(len(sentences) - 1):
                score = util.cos_sim(embeddings[i], embeddings[i + 1]).item()
                if score >= self.threshold:  # 如果相似度高於標準就合起來
                    current_chunk.append(sentences[i + 1])
                else:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) > MAX_CHUNK_SIZE:
                        all_chunks.append(
                            {
                                "page_content": chunk_text,
                                "metadata": metadata.copy(),
                            }
                        )
                    current_chunk = [sentences[i + 1]]

            if current_chunk:
                all_chunks.append(
                    {
                        "page_content": "".join(current_chunk),
                        "metadata": metadata.copy(),
                    }
                )

        return all_chunks


def chunk_documents(docs, language=None, chunk_size=None, chunk_overlap=None):
    chunker = Chunker(model_name="BAAI/bge-m3")
    return chunker.chunk_documents(docs, language)
