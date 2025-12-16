from ollama import Client
from pathlib import Path
import yaml


def load_ollama_config() -> dict:
    """
    讀取設定檔，依照優先順序：
    1. config_local.yaml
    2. config_submit.yaml
    3. 預設值
    """
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

    # 如果都找不到，回傳一個合理的預設值 (針對 Granite)
    if config_path is None:
        return {"host": "http://127.0.0.1:11434", "model": "granite4:3b"}

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return config.get("ollama", {})


def is_contains_chinese(strs):
    """檢查字串是否包含中文字元"""
    for _char in strs:
        if "\u4e00" <= _char <= "\u9fff":
            return True
    return False


def generate_answer(query, context_chunks, ollama_client):
    """
    生成答案的主函數
    Args:
        query: 使用者問題
        context_chunks: 檢索到的文本塊列表
        ollama_client: 從 main.py 傳入的已連線客戶端
    """
    # 1. 準備 Context
    context = "\n\n".join([chunk["page_content"] for chunk in context_chunks])

    # 2. 準備 Prompt
    if is_contains_chinese(query):
        # 【中文 Prompt：強調完整性與無幻覺】
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

    # 3. 設定模型 (比賽指定 Granite)
    # 再次確認：你的 run.sh 或比賽環境必須有這個名稱的模型
    model = "granite4:3b"

    try:
        # num_ctx: 8192 是 Granite 的標準長度，夠用且安全
        # 如果你的硬體記憶體很小，可以改回 4096
        response = ollama_client.generate(
            model=model, prompt=prompt, options={"num_ctx": 8192}
        )
        raw_output = response["response"]
        final_answer = raw_output.strip()

        # 4. 輸出清洗：移除 "Answer:" 或 "回答：" 等開頭贅字
        prefixes = ["Answer:", "回答：", "答案：", "Answer", "回答", "答案"]
        for p in prefixes:
            if final_answer.startswith(p):
                final_answer = final_answer[len(p) :].strip()
            # 處理帶冒號的情況 (例如 "Answer: The...")
            if final_answer.startswith(p + ":") or final_answer.startswith(p + "："):
                final_answer = final_answer[len(p) + 1 :].strip()

        return final_answer

    except Exception as e:
        return f"Error generating answer: {e}"


if __name__ == "__main__":
    # 簡單的本地測試 (如果直接執行這個檔案)
    print("Testing generator.py...")
    # 這裡的測試需要你有本地 Ollama 服務
    try:
        cfg = load_ollama_config()
        client = Client(host=cfg.get("host", "http://localhost:11434"))

        query = "What is the capital of France?"
        chunks = [{"page_content": "France is in Europe. Its capital is Paris."}]

        print(f"Query: {query}")
    except Exception as e:
        print(f"Skipping test due to error: {e}")
