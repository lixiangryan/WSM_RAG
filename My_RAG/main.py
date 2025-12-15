import os
import re
import jieba
from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
from ollama import Client
import argparse

# --- 設定開關 ---
ENABLE_MULTI_QUERY = False

# --- Helper Functions ---


def _split_sentences(text: str, language: str):
    if not text:
        return []
    if language == "zh":
        parts = re.split(r"([。！？])", text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sent = (parts[i] + parts[i + 1]).strip()
            if sent:
                sentences.append(sent)
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        return sentences
    else:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


def _select_reference_sentences(
    query_text: str, retrieved_chunks, language: str, max_refs: int = 10
):
    """從檢索到的 chunks 中挑選最相關的句子作為 references"""
    if not retrieved_chunks:
        return []

    if language == "zh":
        query_tokens = set(token for token in jieba.cut(query_text) if token.strip())
    else:
        word_re = re.compile(r"\w+")
        query_tokens = set(w.lower() for w in word_re.findall(query_text))

    candidate_sentences_with_score = []
    seen_sentences = set()

    # 只看 top 10 chunks
    top_k_chunks = retrieved_chunks[:10]

    for chunk in top_k_chunks:
        # 相容性處理：若 chunk 是字典則取 page_content，若是物件則取屬性
        text = (
            chunk.get("page_content", "")
            if isinstance(chunk, dict)
            else getattr(chunk, "page_content", "")
        )

        for sent in _split_sentences(text, language):
            sent = sent.strip()
            if sent and sent not in seen_sentences:
                score = _calculate_similarity(query_tokens, sent, language)
                candidate_sentences_with_score.append((sent, score))
                seen_sentences.add(sent)

    candidate_sentences_with_score.sort(key=lambda x: x[1], reverse=True)
    references = [item[0] for item in candidate_sentences_with_score[:max_refs]]
    return references


def _calculate_similarity(query_tokens, sent: str, language: str) -> float:
    if not sent.strip():
        return 0.0
    if language == "zh":
        s_tokens = set(token for token in jieba.cut(sent) if token.strip())
    else:
        word_re = re.compile(r"\w+")
        s_tokens = set(w.lower() for w in word_re.findall(sent))

    if not s_tokens:
        return 0.0
    return len(query_tokens & s_tokens) / (len(s_tokens) + 1e-8)


def generate_multiple_queries(original_query: str, ollama_client: Client) -> list[str]:
    """Query Rewriting: 用 LLM 生成多種問法"""
    prompt = f"""You are a helpful assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents. Provide these alternative questions separated by newlines. Only provide the questions, no other text.
Original question: {original_query}"""
    try:
        # 使用較小的模型以節省時間，若無則 fallback
        model_name = os.getenv("REWRITER_MODEL", "gemma:2b")
        response = ollama_client.generate(model=model_name, prompt=prompt, stream=False)
        generated_text = response.get("response", "")
        queries = [q.strip() for q in generated_text.split("\n") if q.strip()]
        queries.insert(0, original_query)
        return list(set(queries))
    except Exception as e:
        print(f"Warning: Failed to generate multiple queries. Error: {e}")
        return [original_query]


# --- MAIN PIPELINE ---


def main(query_path, docs_path, language, output_path):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents (修正：這裡不需要傳 domain，是全量切分)
    print("Chunking documents...")
    chunks = chunk_documents(docs_for_chunking, language)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever (修正：建立檢索器時建立索引)
    print("Creating retriever...")
    retriever_obj = create_retriever(chunks, language)
    print("Retriever created successfully.")

    # --- Setup Ollama Client ---
    hosts_to_try = [
        "http://ollama-gateway:11434",
        "http://ollama:11434",
        "http://localhost:11434",
    ]
    ollama_client = None
    for host in hosts_to_try:
        try:
            temp_client = Client(host=host)
            temp_client.list()
            ollama_client = temp_client
            print(f"Connected to Ollama at {host}")
            break
        except Exception:
            continue

    # 若連不上，這裡會報錯，但在正式環境通常有 fallback
    if ollama_client is None:
        print("Warning: Could not connect to Ollama. Generation might fail.")

    # 4. Processing Queries
    for query in tqdm(queries, desc="Processing Queries"):
        original_query_text = query["query"]["content"]
        qLanguage = query.get("language", language) or "en"

        # --- 關鍵修正：在這裡獲取每個 query 的 domain ---
        # 結構通常是 {"domain": "Finance", "query": {...}}
        query_domain = query.get("domain", None)

        # --- Retrieval Strategy ---
        final_chunks = []

        if ENABLE_MULTI_QUERY and ollama_client:
            # 開啟多重查詢 (較慢)
            all_queries = generate_multiple_queries(original_query_text, ollama_client)
            for q in all_queries:
                # 這裡傳入 query_domain 給 retriever
                final_chunks.extend(
                    retriever_obj.retrieve(q, query_domain=query_domain, top_k=3)
                )

            # 去重 (基於 content)
            unique_chunks = []
            seen_content = set()
            for c in final_chunks:
                content = (
                    c.get("page_content") if isinstance(c, dict) else c.page_content
                )
                if content not in seen_content:
                    unique_chunks.append(c)
                    seen_content.add(content)
            final_chunks = unique_chunks[:10]  # 限制數量

        else:
            # 標準單次查詢 (快速，推薦)
            # 這裡傳入 query_domain，實現分區檢索，解決跨領域雜訊
            final_chunks = retriever_obj.retrieve(
                original_query_text, query_domain=query_domain, top_k=10
            )

        # 5. Generate Answer
        # 注意：generate_answer 需要實作支援 ollama_client 的傳入，若你的 generator.py 沒改，可能要調整
        answer = generate_answer(original_query_text, final_chunks, ollama_client)

        if "prediction" not in query:
            query["prediction"] = {}
        query["prediction"]["content"] = answer

        # 6. Extract References
        reference_sentences = _select_reference_sentences(
            original_query_text, final_chunks, qLanguage, max_refs=10
        )
        query["prediction"]["references"] = reference_sentences

    # 7. Save Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_jsonl(output_path, queries)
    print(f"Predictions saved at '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", help="Path to the query file")
    parser.add_argument("--docs_path", help="Path to the documents file")
    parser.add_argument(
        "--language",
        help="Language to filter queries (zh or en), if not specified, process all",
    )
    parser.add_argument("--output", help="Path to the output file")
    args = parser.parse_args()

    main(args.query_path, args.docs_path, args.language, args.output)
