import yaml
import re
from pathlib import Path
from ollama import Client

# ==========================================
# 1. Config & Connection Helpers
# ==========================================


def load_ollama_config() -> dict:
    """讀取設定檔，決定使用哪個模型"""
    configs_folder = Path(__file__).parent.parent / "configs"
    config_paths = [
        configs_folder / "config_local.yaml",
        configs_folder / "config_submit.yaml",
    ]
    config = {}
    for path in config_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            config = raw.get("ollama", {})
            break

    # 預設值 (Fallback)
    if not config:
        config = {
            "host": "http://127.0.0.1:11434",
            "model": "granite3-dense:8b",  # 建議換成你實際在跑的模型名稱
        }

    # 確保有 model key
    if "model" not in config:
        config["model"] = "granite3-dense:8b"

    return config


def get_fallback_client() -> Client:
    """
    僅當 main.py 沒傳 client 進來時，才使用的備用連線方式。
    """
    hosts = [
        "http://ollama-gateway:11434",
        "http://ollama:11434",
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ]
    for host in hosts:
        try:
            client = Client(host=host)
            client.list()
            print(f"[Generator] Fallback connected to {host}")
            return client
        except Exception:
            continue
    raise ConnectionError("Generator failed to connect to any Ollama host.")


# ==========================================
# 2. Parsing Logic (關鍵：分離思考與答案)
# ==========================================


def is_contains_chinese(strs):
    for _char in strs:
        if "\u4e00" <= _char <= "\u9fff":
            return True
    return False


def _parse_model_output(response_text: str) -> str:
    """
    解析模型輸出，移除 <Thinking> 區塊，只保留最終答案。
    支援的標籤： 'Final Answer:', '最終答案：', 'Answer:', '回答：'
    """
    text = response_text.strip()

    # 定義分割關鍵字 (優先順序很重要)
    split_markers = [
        "Final Answer:",
        "Final Answer",
        "最终答案：",
        "最终答案:",
        "最终答案",  # 簡體
        "最終答案：",
        "最終答案:",
        "最終答案",  # 繁體
        "Answer:",
        "Answer",
        "回答：",
        "回答:",
        "回答",
    ]

    # 嘗試分割
    for marker in split_markers:
        if marker in text:
            # 取最後一個出現的 marker 之後的內容 (避免思考過程中提到這些詞)
            parts = text.split(marker)
            if len(parts) > 1:
                answer_part = parts[-1].strip()
                # 進一步清理開頭的冒號或符號
                if answer_part.startswith(":") or answer_part.startswith("："):
                    answer_part = answer_part[1:].strip()
                return answer_part

    # 如果都沒找到 marker，嘗試用 Regex 移除 <think>...</think> 標籤 (DeepSeek 風格)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return text


# ==========================================
# 3. Generator Function (核心)
# ==========================================


def generate_answer(query: str, context_chunks: list, client: Client = None) -> str:
    """
    Args:
        query: 使用者問題
        context_chunks: 檢索到的 chunks 列表
        client: 由 main.py 傳入的 Ollama Client (必填，但保留 None 做 fallback)
    """
    # 1. 處理 Context
    if not context_chunks:
        return "I don't know" if not is_contains_chinese(query) else "我不知道"

    context_text = "\n\n".join(
        [
            f"[Document {i+1}]: {chunk['page_content']}"
            for i, chunk in enumerate(context_chunks)
        ]
    )

    # 2. 構建 Prompt (加入 CoT 思維鏈)
    # 這是提升準確度、減少幻覺的最強手段
    if is_contains_chinese(query):
        prompt = (
            "你是一個專業的 RAG 助手。請嚴格根據下方的【參考文件】回答使用者的問題。\n"
            "規則：\n"
            "1. 若【參考文件】中沒有答案，請直接回答「我不知道」，不可編造。\n"
            "2. 嚴格比對公司名稱、時間點（年份/月份）與事件。若年份不符，視為無效資訊。\n"
            "3. 回答必須簡潔有力。\n"
            "4. 請先在心中思考，然後輸出最終答案。\n\n"
            "輸出格式：\n"
            "思考過程：(簡述你的推論邏輯，檢查年份與公司名稱)\n"
            "最終答案：(只輸出結論)\n\n"
            f"【參考文件】：\n{context_text}\n\n"
            f"【使用者問題】：{query}\n"
        )
    else:
        prompt = (
            "You are a strict RAG assistant. Answer the question based ONLY on the provided context.\n"
            "Rules:\n"
            "1. If the answer is not in the context, say 'I don't know'. Do not hallucinate.\n"
            "2. Strictly verify company names, dates (year/month), and events. Mismatches are invalid.\n"
            "3. Be concise.\n"
            "4. Think step-by-step before answering.\n\n"
            "Output Format:\n"
            "Thinking: (Brief reasoning, verifying dates and entities)\n"
            "Final Answer: (The conclusion only)\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
        )

    # 3. 準備 Client 與 Model
    cfg = load_ollama_config()
    model = cfg.get("model", "granite4:3b")  # 注意：這裡預設值要看你實際跑什麼

    if client is None:
        try:
            client = get_fallback_client()
        except Exception as e:
            return f"Error: {str(e)}"

    # 4. 生成回答
    try:
        response = client.generate(model=model, prompt=prompt, stream=False)
        raw_output = response.get("response", "")

        # 5. 解析回答 (去除思考過程)
        final_answer = _parse_model_output(raw_output)

        # 安全網：如果解析後為空，至少回傳原始輸出 (雖然可能會髒一點)
        if not final_answer:
            return raw_output

        return final_answer

    except Exception as e:
        print(f"[Generator Error] {e}")
        return "Generation failed."


# ==========================================
# 4. Local Test (測試區)
# ==========================================
if __name__ == "__main__":
    # 模擬測試
    print("Testing Generator...")

    # 模擬資料
    mock_chunks = [
        {"page_content": "2018年，CleanCo 營收為 500萬。"},
        {"page_content": "2020年，Retail Emporium 營收為 4.8億。"},
    ]

    # 嘗試建立一個本地 client 進行測試
    try:
        test_client = Client(host="http://localhost:11434")

        # 測試比較題
        q = "比較 CleanCo 2018 和 Retail Emporium 2020 的營收，誰比較高？"
        print(f"\nQuestion: {q}")
        ans = generate_answer(q, mock_chunks, test_client)
        print(f"Result: {ans}")

    except Exception as e:
        print(f"Skipping test, no local ollama found: {e}")
