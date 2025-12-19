import sys
import os
import json
import logging
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from My_RAG.chunker import chunk_documents
from My_RAG.knowledge_graph import SimpleKnowledgeGraph

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DOCS_PATH = os.path.join("dragonball_dataset", "dragonball_docs.jsonl")

def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph Index (v3.0 Global Co-occurrence)")
    parser.add_argument("--language", type=str, required=True, choices=["en", "zh"], help="Target language (en or zh)")
    args = parser.parse_args()
    
    target_lang = args.language
    output_filename = f"kg_index_{target_lang}.json"
    
    logging.info(f"--- Building KG Index for [{target_lang}] ---")

    # 1. Load Docs
    docs_list = []
    try:
        if not os.path.exists(DOCS_PATH):
            raise FileNotFoundError(f"Documents file not found: {DOCS_PATH}")
        
        logging.info("Loading documents...")
        with open(DOCS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs_list.append(json.loads(line))
        logging.info(f"Loaded {len(docs_list)} total documents.")
        
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        sys.exit(1)

    # 2. Chunking
    # Must match Runtime configuration exactly
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 300 # Baseline config
    
    logging.info(f"Chunking documents (Lang={target_lang}, Size={CHUNK_SIZE})...")
    chunks = chunk_documents(docs_list, language=target_lang, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logging.info(f"Generated {len(chunks)} chunks.")

    # 3. Build Graph
    logging.info("Initializing Knowledge Graph (This builds Entity & Co-occurrence Maps)...")
    kg = SimpleKnowledgeGraph(chunks) # index_path=None -> Forces build from scratch

    # 4. Save
    logging.info(f"Extracted {len(kg.entity_map)} entities.")
    logging.info(f"Saving index v3.0 to {output_filename}...")
    kg.save(output_filename)
    logging.info("Done.")

if __name__ == "__main__":
    main()
