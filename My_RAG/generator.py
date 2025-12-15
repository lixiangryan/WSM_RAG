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


def is_contains_chinese(strs):
    for _char in strs:
        if "\u4e00" <= _char <= "\u9fff":
            return True
    return False


def generate_answer(query, context_chunks, ollama_client):
    context = "\n\n".join([chunk["page_content"] for chunk in context_chunks])

    if is_contains_chinese(query):
        # 【中文 Prompt：強調完整性與無幻覺】
        # 移除了 "max 3 sentences" 和 "Thinking"
        prompt = (
            "你是一個專業的問答助手。請「完全依據」以下的參考內容來回答使用者的問題。\n"
            "若參考內容中沒有答案，請直接回答「我不知道」，絕對不要編造內容。\n\n"
            "回答原則：\n"
            "1. 答案必須詳盡且完整，不要遺漏關鍵數據（如年份、公司名、具體數字）。\n"
            "2. 直接給出答案，不需要解釋你的思考過程。\n"
            "3. 請使用繁體中文回答。\n\n"
            f"參考內容 (Context):\n{context}\n\n"
            f"使用者問題 (Question): {query}\n\n"
            "回答 (Answer):"
        )
    else:
        # 【英文 Prompt】
        prompt = (
            "You are a helpful assistant. Answer the question based ONLY on the provided context.\n"
            "If the answer is not in the context, say 'I don't know'. Do not hallucinate.\n\n"
            "Guidelines:\n"
            "1. Be comprehensive and include all key details (years, names, figures).\n"
            "2. Provide the answer directly without explaining your reasoning.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    model = "granite4:3b"
    try:
        response = ollama_client.generate(
            model=model, prompt=prompt, options={"num_ctx": 16384}
        )
        raw_output = response["response"]
        final_answer = raw_output.strip()

        # 簡單清洗：如果開頭有 "Answer:" 或 "回答：" 就切掉
        prefixes = ["Answer:", "回答：", "答案：", "Answer", "回答", "答案"]
        for p in prefixes:
            if final_answer.startswith(p):
                final_answer = final_answer[len(p) :].strip()
            # 處理帶冒號的情況
            if final_answer.startswith(p + ":") or final_answer.startswith(p + "："):
                final_answer = final_answer[len(p) + 1 :].strip()

        return final_answer

    except Exception as e:
        return f"Error generating answer: {e}"


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {
            "page_content": "The Eiffel Tower is located in Paris, the capital city of France."
        },
    ]
    # Mock client for testing if needed, or just comment out
    # answer = generate_answer(query, context_chunks, client)
    # print("Generated Answer:", answer)
    pass
