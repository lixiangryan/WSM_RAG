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


# --- MAIN FUNCTION ---
# --- 讓references只提取最重要的sentence，不要整個chunk塞進去 ---
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


# --- Selects the most relevant sentences from retrieved chunks as references. ---
def _select_reference_sentences(
    query_text: str, retrieved_chunks, language: str, max_refs: int = 10
):
    if not retrieved_chunks:
        return []

    # --- 預處理 Query Tokens ---
    if language == "zh":
        query_tokens = set(token for token in jieba.cut(query_text) if token.strip())
    else:
        word_re = re.compile(r"\w+")
        query_tokens = set(w.lower() for w in word_re.findall(query_text))

    candidate_sentences_with_score = []
    seen_sentences = set()

    # 只對top10以內的chunk去做執行
    top_k_chunks = retrieved_chunks[:10]

    for chunk in top_k_chunks:
        text = chunk.get("page_content", "")

        for sent in _split_sentences(text, language):
            sent = sent.strip()

            if sent and sent not in seen_sentences:
                score = _calculate_similarity(query_tokens, sent, language)
                candidate_sentences_with_score.append((sent, score))
                seen_sentences.add(sent)

    # --- sentences排序 ---
    candidate_sentences_with_score.sort(key=lambda x: x[1], reverse=True)

    # 取出前 max_refs
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


# --- 將「一個問題」，透過 LLM 擴寫成「多個不同問法」，藉此增加在資料庫中撈到正確文章的機率 ---
def generate_multiple_queries(original_query: str, ollama_client: Client) -> list[str]:
    prompt = f"""You are a helpful assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents. Provide these alternative questions separated by newlines. Only provide the questions, no other text.
Original question: {original_query}"""
    try:
        model_name = os.getenv("REWRITER_MODEL", "granite4:3b")
        response = ollama_client.generate(model=model_name, prompt=prompt, stream=False)
        generated_text = response.get("response", "")
        queries = [q.strip() for q in generated_text.split("\n") if q.strip()]
        queries.insert(0, original_query)
        return list(set(queries))
    except Exception as e:
        print(
            f"Warning: Failed to generate multiple queries, using original query only. Error: {e}"
        )
        return [original_query]


# --- MAIN  ---
def main(query_path, docs_path, language, output_path):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    chunks = chunk_documents(docs_for_chunking, language)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")

    # --- Ollama Client Instantiation ---
    hosts_to_try = [
        "http://ollama-gateway:11434",  # Submission host
        "http://ollama:11434",  # Local Docker host
        "http://localhost:11434",  # Local Conda host
    ]
    ollama_client = None
    for host in hosts_to_try:
        try:
            temp_client = Client(host=host)
            temp_client.list()  # Test connectivity
            ollama_client = temp_client
            print(f"Connected to Ollama at {host}")
            break  # Successfully connected, exit loop
        except Exception as e:
            print(
                f"Warning: Failed to connect to Ollama at {host}. Trying next host. Error: {e}"
            )
            continue

    if ollama_client is None:
        raise ConnectionError("Failed to connect to any Ollama host.")
    # --- End Ollama Client Instantiation ---

    retriever = create_retriever(chunks, language, ollama_client)
    print("Retriever created successfully.")

    for query in tqdm(queries, desc="Processing Queries"):
        original_query_text = query["query"]["content"]
        qLanguage = query.get("language", language) or "en"

        # --- 原本的問題改寫成 3 個不同的問法 ---
        all_queries = generate_multiple_queries(original_query_text, ollama_client)

        retrieved_chunks_list = []
        for q in all_queries:
            retrieved_chunks_list.extend(
                retriever.retrieve(q, top_k=3)
            )  # Retrieve top 3 for each query

        final_chunks = retrieved_chunks_list
        # --- End Multi-Query Enabled ---

        # 5. Generate Answer
        # Use original query for answer generation
        answer = generate_answer(original_query_text, final_chunks, ollama_client)

        if "prediction" not in query:
            query["prediction"] = {}
        query["prediction"]["content"] = answer

        # Use the same chunks for reference selection
        reference_sentences = _select_reference_sentences(
            original_query_text, final_chunks, qLanguage, max_refs=10
        )
        query["prediction"]["references"] = reference_sentences

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
