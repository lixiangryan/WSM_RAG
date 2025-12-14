import sys
import os
import json
import logging
import argparse
from collections import defaultdict
from tqdm import tqdm
from ollama import Client

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_kg_index(lang):
    filename = f"kg_index_{lang}.json"
    if not os.path.exists(filename):
        logging.error(f"KG Index not found: {filename}. Please run build_kg_index.py first.")
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_synonyms(entities, client, model="granite4:3b"):
    """
    Uses LLM to group entities into synonyms.
    Strategy:
    1. Sort entities by frequency.
    2. Take top N entities.
    3. Ask LLM to group them.
    """
    # Filter only Terms (ignore Years)
    terms = [e for e in entities if e.startswith("Term:")]
    # Extract raw text
    raw_terms = [t.split(":", 1)[1] for t in terms]
    
    # We can't send all terms at once. Let's send batches of potentially related terms?
    # Or just send the top 100 terms and ask for clusters.
    
    top_terms = raw_terms[:200] # Take top 200 most frequent terms
    
    prompt = f"""Group the following terms into synonym clusters.
Identify terms that refer to the same entity (e.g., "TSMC", "Taiwan Semiconductor", "台積電").
Output a JSON object where the key is the canonical term (the most standard one) and the value is a list of synonyms.
Example: {{"TSMC": ["Taiwan Semiconductor", "台積電"], "Revenue": ["Sales", "Turnover"]}}

Terms:
{", ".join(top_terms)}

JSON Output:"""

    try:
        logging.info("Sending request to LLM for synonym clustering...")
        response = client.generate(model=model, prompt=prompt, stream=False, format="json")
        content = response.get("response", "")
        
        clusters = json.loads(content)
        
        # Invert to Synonym -> Canonical
        synonym_map = {}
        for canonical, variants in clusters.items():
            # Canonical maps to itself (optional, but good for consistency)
            # synonym_map[canonical] = canonical 
            for v in variants:
                if v != canonical:
                    synonym_map[v] = canonical
                    
        return synonym_map
        
    except Exception as e:
        logging.error(f"LLM generation failed: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Build Synonym Map for KG (Phase 6)")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"], help="Target language")
    args = parser.parse_args()
    
    # 1. Load KG Index to get entities and frequencies
    kg_data = load_kg_index(args.language)
    if not kg_data:
        return

    doc_freqs = kg_data.get("doc_freqs", {})
    # Sort by frequency
    sorted_entities = sorted(doc_freqs.keys(), key=lambda k: doc_freqs[k], reverse=True)
    
    logging.info(f"Loaded {len(sorted_entities)} entities from KG.")
    
    # 2. Connect to Ollama
    try:
        client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        client.list()
    except Exception as e:
        logging.error(f"Ollama connection failed: {e}")
        return

    # 3. Generate Synonyms
    synonym_map = generate_synonyms(sorted_entities, client)
    
    logging.info(f"Generated {len(synonym_map)} synonym mappings.")
    
    # 4. Save
    output_file = "synonym_map.json"
    
    # Merge with existing if any
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        existing.update(synonym_map)
        synonym_map = existing
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(synonym_map, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Saved synonym map to {output_file}")

if __name__ == "__main__":
    main()
