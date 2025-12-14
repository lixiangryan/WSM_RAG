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

# --- [NEW] Sentence Selection and Multi-Query Functions ---
def _split_sentences(text: str, language: str):
    """Very simple sentence splitter for zh / en."""
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

def _select_reference_sentences(query_text: str, retrieved_chunks, language: str, max_refs: int = 5):
    """Selects the most relevant sentences from retrieved chunks as references."""
    if not retrieved_chunks:
        return []
    candidate_sentences = []
    for chunk in retrieved_chunks:
        text = chunk.get("page_content", "")
        for sent in _split_sentences(text, language):
            if sent.strip():
                candidate_sentences.append(sent.strip())
    if not candidate_sentences:
        return []

    if language == "zh":
        q_tokens = set(token for token in jieba.cut(query_text) if token.strip())
        def sent_score(sent: str):
            s_tokens = set(token for token in jieba.cut(sent) if token.strip())
            if not s_tokens: return 0.0
            return len(q_tokens & s_tokens) / (len(s_tokens) + 1e-8)
    else:
        word_re = re.compile(r"\w+")
        q_tokens = set(w.lower() for w in word_re.findall(query_text))
        def sent_score(sent: str):
            s_tokens = set(w.lower() for w in word_re.findall(sent))
            if not s_tokens: return 0.0
            return len(q_tokens & s_tokens) / (len(s_tokens) + 1e-8)

    scored = sorted([(sent, sent_score(sent)) for sent in candidate_sentences], key=lambda x: x[1], reverse=True)
    references = [sent for sent, score in scored if sent not in locals().get('references', [])][:max_refs]
    return references

def generate_multiple_queries(original_query: str, ollama_client: Client) -> list[str]:
    """Generates variations of the original query using an LLM."""
    prompt = f"""You are a helpful assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents. Provide these alternative questions separated by newlines. Only provide the questions, no other text.
Original question: {original_query}"""
    try:
        # Optimization: Use same model as other tasks to avoid VRAM swapping thrashing
        model_name = os.getenv("REWRITER_MODEL", "granite4:3b")
        response = ollama_client.generate(model=model_name, prompt=prompt, stream=False)
        generated_text = response.get("response", "")
        queries = [q.strip() for q in generated_text.split('\n') if q.strip()]
        queries.insert(0, original_query)
        return list(set(queries))
    except Exception as e:
        print(f"Warning: Failed to generate multiple queries, using original query only. Error: {e}")
        return [original_query]

def main(query_path, docs_path, language, output_path):
    print("Loading documents and queries...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    
    print("Connecting to Ollama...")
    hosts_to_try = ["http://ollama-gateway:11434", "http://ollama:11434", "http://localhost:11434"]
    ollama_client = None
    for host in hosts_to_try:
        try:
            temp_client = Client(host=host)
            temp_client.list()
            ollama_client = temp_client
            print(f"Connected to Ollama at {host}")
            break
        except Exception as e:
            print(f"Warning: Failed to connect to Ollama at {host}. Trying next host.")
    if ollama_client is None:
        raise ConnectionError("Failed to connect to any Ollama host.")

    print("Chunking documents...")
    # Pass ollama_client to chunker for fallback
    chunks = chunk_documents(docs_for_chunking, language, ollama_client=ollama_client)
    
    print("Creating retriever...")
    index_filename = f"kg_index_{language}.json" if language else "kg_index.json"
    retriever = create_retriever(chunks, language, index_path=index_filename)
    
    for query in tqdm(queries, desc="Processing Queries"):
        original_query_text = query['query']['content']
        qLanguage = query.get("language", language) or "en"
        
        # --- Multi-Query Enabled for Phase 2 Experiment ---
        # 4a. Generate multiple queries
        all_queries = generate_multiple_queries(original_query_text, ollama_client)
        
        # 4b. Retrieve for all generated queries
        # The retriever expects a single query. We need to collect results from all queries.
        retrieved_chunks_list = []
        for q in all_queries:
            retrieved_chunks_list.extend(retriever.retrieve(q, top_k=10, use_hyde=False)) # Retrieve top 10 for each query

        # Deduplicate chunks to avoid redundant processing, though their scores might be different
        # For this experiment, we'll just combine them and assume the generator handles duplicates.
        final_chunks = retrieved_chunks_list
        # --- End Multi-Query Enabled ---

        # 5. Generate Answer
        # Use original query for answer generation
        answer = generate_answer(original_query_text, final_chunks, ollama_client)

        if "prediction" not in query:
            query["prediction"] = {}
        query["prediction"]["content"] = answer
        
        # Use the same chunks for reference selection
        reference_sentences = _select_reference_sentences(original_query_text, final_chunks, qLanguage, max_refs=5)
        query["prediction"]["references"] = reference_sentences

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_jsonl(output_path, queries)
    print(f"Predictions saved at '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en)')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
