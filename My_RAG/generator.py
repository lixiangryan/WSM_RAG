from ollama import Client
from pathlib import Path
import yaml

def load_ollama_config() -> dict:
    configs_folder = Path(__file__).parent.parent / "configs"
    config_paths = [
        configs_folder / "config_local.yaml",
        configs_folder / "config_submit.yaml",
    ]
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError("No configuration file found.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]


def generate_answer(query, context_chunks, ollama_client):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    #print("wwwwwwwwwwwwwwwwww_Context for answer generation:", context)
    prompt = f"""You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\n\nQuestion: {query} \nContext: {context} \nAnswer:\n"""
    
    model="granite4:3b"
    try:
        response = ollama_client.generate(
            model=model, 
            prompt=prompt,
            options={"num_ctx": 16384}
            )
        return response["response"]
    except Exception as e:
        return f"Error generating answer: {e}"

if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    # Mock client for testing if needed, or just comment out
    # answer = generate_answer(query, context_chunks, client)
    # print("Generated Answer:", answer)
    pass