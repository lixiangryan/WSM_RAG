# WSM RAG 專案評估指南

本專案使用 `docker-compose` 建立一個可攜式、自包含的 RAG（Retrieval-Augmented Generation）評估環境。它包含兩個服務：
1.  **`app`**：執行主要 RAG 程式 (`My_RAG`) 和評估工具 (`rageval`) 的 Python 服務。
2.  **`ollama`**：一個專門用來跑「裁判模型 (Judge LLM)」的服務，`rageval` 會呼叫它來為 RAG 答案評分。

使用 `docker-compose` 可以確保所有協作者都有一致的執行環境，無需在本機手動安裝 Ollama 或處理複雜的網路設定。

## 執行需求 (Prerequisites)

* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* 一台有 **NVIDIA GPU** 的主機 (本專案已針對 6GB VRAM 進行優化)
* 確保 Docker Desktop 已設定使用 NVIDIA GPU (通常是預設)

## ⚡ 如何執行 (How to Run)

**只需要這一個指令。**

在專案的根目錄 (包含 `docker-compose.yml` 的地方)，打開你的終端機 (PowerShell / Terminal) 並執行：

```bash
docker-compose up --build
```

### 第一次執行會發生什麼？

1.  **Build Services:** `docker-compose` 會使用 `Dockerfile` 和 `ollama_service/Dockerfile` 分別建立 `app` 和 `ollama_service` 的映像。這個過程現在非常快，因為模型不會在 build 階段下載。
2.  **Start Services:** 啟動 `ollama_service` 和 `app` 兩個 container。
3.  **Run `entrypoint.sh` & Download Model:**
    *   `ollama_service` 容器啟動後，會執行 `entrypoint.sh` 腳本。
    *   此腳本會先啟動 Ollama 伺服器，然後**自動下載 `gemma:2b` 模型** (約 2.5GB)。
    *   模型會被下載到 `ollama_storage` 這個 Volume 中，這意味著**未來再啟動就無需重新下載**。
    *   **(請在此時保持耐心，這只需要一次)**
4.  **Run `run.sh`:** 當 `ollama_service` 準備就緒後，`app` 服務會自動開始執行 `run.sh` 腳本 (包含 RAG 預測和評估)。
5.  **Run Evaluation:** `ollama_service` 會將模型載入 VRAM，`rageval` 的評估進度條就會開始跑了。

### 未來執行

1. **重新跑一次評估 (無修改程式碼):**
   ```bash
   docker-compose up
   ```

2. **重新建置並執行 (有修改程式碼):**
   ```bash
   docker-compose up --build --force-recreate
   ```

3. **跑完自動關閉並清除所有服務:**
   ```bash
   docker-compose up --build --force-recreate --abort-on-container-exit
   ```

## 📜 專案配置修改總結

為了讓專案能在有限的硬體資源上穩定運行、方便協作並解決模型持久化問題，我們做了以下關鍵修改：

### 1. `docker-compose.yml` (修改)

*   **目的：** 用來管理 `app` 和 `ollama_service` 兩個服務。
*   **配置：**
    *   `ollama_service` 服務：改為由本地 `ollama_service/Dockerfile` build 而來。
    *   `volumes`: 為 `ollama_service` 建立一個永久儲存卷 (`ollama_storage`) 來存放下載的模型，確保模型在容器重啟後依然存在。

### 2. `ollama_service/entrypoint.sh` (新增)

*   **目的：** 將模型下載從「建構階段」移至「執行階段」，確保模型被下載到掛載的 Volume 中。
*   **策略：**
    1.  在容器啟動時，先在背景執行 `ollama serve`。
    2.  輪詢偵測，直到 Ollama 伺服器完全就緒。
    3.  執行 `ollama pull gemma:2b`，將模型下載到 `ollama_storage` Volume。
    4.  保持容器運行，讓 `app` 服務可以連接。

### 3. `ollama_service/Dockerfile` (修改)

*   **目的：** 設定 `ollama_service` 的啟動行為，並確保必要的工具已安裝。
*   **修改：**
    *   **新增 `RUN apt-get update && apt-get install -y curl`**：手動安裝 `curl` 工具，因為 `entrypoint.sh` 腳本需要它來檢查 Ollama 服務狀態。
    *   移除在 build 階段執行 `RUN ollama pull ...` 的指令。
    *   改為複製 `entrypoint.sh` 腳本到映像中，並將其設為 `ENTRYPOINT`。

### 4. Dockerfile (修改)

*   **CRLF & BOM 修正：** 加入 `sed` 指令來自動修正 Windows 的換行符號 (`\r`) 和 UTF-8 BOM，解決 `exec format error`。具體來說，針對 `run.sh` 和 `wait_and_run.sh` 腳本，新增以下 `sed` 指令：
    ```dockerfile
    # --- [修正: 移除 Windows 換行符號 (CRLF) 和 BOM] ---
    RUN sed -i 's/\r$//' run.sh
    RUN sed -i '1s/^\xEF\xBB\xBF//' run.sh
    # [新增] 修正 wait_and_run.sh 的換行符號
    RUN sed -i 's/\r$//' wait_and_run.sh
    RUN sed -i '1s/^\xEF\xBB\xBF//' wait_and_run.sh
    ```
*   **`CMD` 修正：** 將 `CMD` 從 `["./run.sh"]` 修改為 `["/bin/sh", "./run.sh"]`，以正確執行沒有 shebang (`#!/bin/sh`) 的腳本。

### 5. `rageval/evaluation/main.py` (修改)

*   **目的：** 替換掉極度消耗資源的預設「裁判模型」。
*   **修改：**
    *   **舊：** `process_jsonl(..., "llama3.3:70b", "v1")`
    *   **新：** `process_jsonl(..., "gemma:2b", "v1")`
*   **原因：** `llama3.3:70b` 需要 40GB+ VRAM，而 `gemma:2b` 僅需 ~3GB VRAM，非常適合在 6GB VRAM 的硬體上進行輕量級評估。

### 6. `rageval/evaluation/metrics/rag_metrics/keypoint_metrics.py` (修改)

*   **目的：** 修正 Docker 內部的網路連線問題。
*   **修改：**
    *   **舊：** `base_url="http://localhost:11434/v1"`
    *   **新：** `base_url="http://ollama:11434/v1"`
*   **原因：** 在 `docker-compose` 網路中，`app` 服務必須使用 `ollama` 服務的「服務名稱」(`ollama`) 來連接，而不是 `localhost`。

### 7. 1201 - 符合提交格式要求 (1201 - Align with Submission Format)

#### a. 調整資料集以符合格式要求

*   **目的：** 使用新的 `queries_show` 資料集進行預測。此資料集不包含 `ground_truth`，專門用於生成展示用的答案。
*   **修改 `run.sh`：**
    *   將 RAG 系統 (`My_RAG/main.py`) 的 `--query_path` 參數，從舊的 `test_queries_*.jsonl` 檔案，改為指向新的 `dragonball_dataset/queries_show/queries_*.jsonl`。
    *   由於新資料集沒有 `ground_truth` 會導致評估階段出錯，因此將執行 `rageval` 評估腳本的指令註解掉。

#### b. 修正 Docker 內部 Ollama 連線問題

*   **目的：** 解決在 Docker 環境中，`app` 服務無法連線到 `ollama` 服務的問題。
*   **修改 `My_RAG/generator.py`：**
    *   **舊：** `client = Client()`，此初始化方式會預設連線到 `localhost`，在容器內部會找不到目標服務。
    *   **新：** `client = Client(host='http://ollama:11434')`，明確指定連線到 `docker-compose.yml` 中定義的 `ollama` 服務名稱。

## 疑難排解 (Troubleshooting)

### 1. `ollama-1 | exec /entrypoint.sh: no such file or directory`

**問題描述：**
當 `ollama_service` 容器啟動時，可能會遇到 `exec /entrypoint.sh: no such file or directory` 錯誤，導致 Ollama 服務無法啟動。這通常發生在 `entrypoint.sh` 腳本是在 Windows 環境下建立或編輯，導致其包含 Windows 風格的換行符號 (CRLF)，而 Docker 容器內部的 Linux 環境無法正確解析。

**解決方案：**
在 `ollama_service/Dockerfile` 中，於 `COPY entrypoint.sh /entrypoint.sh` 之後，加入一行 `RUN sed -i 's/\r$//' /entrypoint.sh`。這會將腳本中的所有 CRLF 換行符號轉換為 Linux 兼容的 LF 換行符號。

**修改範例 (ollama_service/Dockerfile):**
```dockerfile
# ... (其他指令)
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh # 新增此行
RUN chmod +x /entrypoint.sh
# ... (其他指令)
```

### 2. `app-1 | ./wait_and_run.sh: 7: curl: not found` (**v0.1 更新**)

**問題描述：**
`app` 容器在執行 `wait_and_run.sh` 腳本時，可能會報告 `curl: not found` 錯誤。這是因為 `app` 服務的基礎映像 `python:3.9-slim` 是一個輕量級映像，預設沒有安裝 `curl` 工具，而 `wait_and_run.sh` 腳本需要 `curl` 來檢查 Ollama 服務的健康狀態。

**解決方案：**
在主 `Dockerfile` 中，於 `WORKDIR /app` 之後，加入一行 `RUN apt-get update && apt-get install -y curl`。這會在建置 `app` 容器時安裝 `curl`。

**修改範例 (Dockerfile):**
```dockerfile
# ... (其他指令)
WORKDIR /app
RUN apt-get update && apt-get install -y curl # 新增此行
# ... (其他指令)
```

**執行 `docker-compose up --build --force-recreate` 重新建置並啟動服務以應用這些修復。**

## lixiang_1114當前任務 (保留，此為專案原有任務記錄)
紀錄每次結果並加上時間戳記

## 結果在這裡
執行 `run.sh` 腳本後，結果將會儲存在專案根目錄下的 `results` 和 `predictions` 資料夾中。這兩個資料夾都會包含以時間戳記命名的子資料夾，例如 `results/20251114_074809/` 和 `predictions/20251114_074809/`。

1.  **`result` 資料夾：**
    *   包含最終的「評估分數」檔案。
    *   例如：`results/YYYYMMDD_HHMMSS/score_en.jsonl` 和 `results/YYYYMMDD_HHMMSS/score_zh.jsonl`。
    *   這些檔案包含了 RAG 評估的最終分數。

2.  **`predictions` 資料夾：**
    *   包含中間的「RAG 預測」檔案。
    *   例如：`predictions/YYYYMMDD_HHMMSS/predictions_en.jsonl` 和 `predictions/YYYYMMDD_HHMMSS/predictions_zh.jsonl`。
    *   這些是 `rageval` 讀取並用來計算分數的原始答案。

## 新查詢資料集格式說明 (New Query Dataset Format Description)

在 `dragonball_dataset/queries_show` 中的查詢檔案 (例如 `queries_en.jsonl`) 具有以下特點：

*   **`prediction.content`**：
    *   此欄位預期在程式執行過程中由 RAG 系統填入對應查詢的答案。
    *   在執行前，此欄位通常會是空的字串。
    *   類型：`str`
*   **`prediction.references`**：
    *   此欄位預期在程式執行過程中由 RAG 系統填入與查詢相關的文件片段（來自 `dragonball_docs.jsonl`）。
    *   在執行前，此欄位通常會是空的列表。
    *   類型：`list[str]`

## 🧹 如何停止與清理

1.  **停止服務：** 在 `docker-compose up` 正在運行的終端機中，按下 `Ctrl + C`。
2.  **停止並移除 Container：** (如果服務是在背景 `-d` 執行，或你想徹底清理)
    ```bash
    docker-compose down
    ```
3.  **移除 Ollama 模型快取 (非必要)：** 如果你想刪除下載的 `gemma:2b` 模型，執行：
    ```bash
    docker-compose down -v
    ```
    (`-v` 會連同 `ollama_storage` volume 一起刪除)

## 🚀 優化 (Optimization)

### lixiang1201_2323_optimize-rag-performance

**目標：** 優化 Ollama 客戶端 (Client) 的實例化，以提高 RAG 管道的整體執行效能。

**改動內容：**

1.  **`My_RAG/main.py` (修改):**
    *   將 `ollama.Client` 的實例化邏輯和主機備援 (fallback) 邏輯從 `My_RAG/generator.py` 移至 `main.py`。
    *   此邏輯現在位於查詢處理迴圈之前，確保 `ollama.Client` 物件只會被建立一次。
    *   在 `main.py` 中，新增了 `ollama.Client` 和 `os` 的 import。
    *   `generate_answer` 函式的呼叫現在會傳遞這個已經實例化好的 `ollama_client` 物件。
    *   新增了連線失敗時的錯誤處理，如果所有主機都無法連線，會拋出 `ConnectionError`。

2.  **`My_RAG/generator.py` (修改):**
    *   移除了 `from ollama import Client` 的 import (因為 client 會從 `main.py` 傳入)。
    *   修改了 `generate_answer` 函式的簽名 (signature)，使其接受一個 `ollama_client` 物件作為參數。
    *   移除了函式內部重複建立 `ollama.Client` 物件和主機備援的邏輯。
    *   直接使用傳入的 `ollama_client` 物件來進行 `generate` 操作。

**優化效益：**
*   避免了在每個查詢中重複實例化 `ollama.Client` 物件和執行連線檢查，大幅減少了不必要的開銷。
*   提高了 RAG 管道在處理大量查詢時的執行效率。

### esdese-feature-1
**目標：** 引入混合檢索 (Hybrid Retrieval) 與重排序 (Reranking) 機制，提升檢索準確度。

**改動內容：**

1.  **Hybrid Retrieval (混合檢索):**
    *   結合 **BM25** (關鍵字檢索) 與 **Embedding Similarity** (向量語意檢索)。
    *   使用 `ollama` 的 embedding 模型 (或 `SentenceTransformer`) 生成文件與查詢的向量。
    *   透過加權平均 (`alpha` 參數) 合併兩者的分數，兼顧精確匹配與語意理解。

2.  **LLM Reranker (重排序):**
    *   在初步檢索出候選文件後，使用 LLM (如 `granite4:3b`) 對候選文件進行再一次的相關性評分。
    *   根據 LLM 的評分對結果進行重新排序，將最相關的文件排在最前面。

3.  **整合優化:**
    *   將上述功能整合進 `My_RAG/retriever.py`。
    *   配合 `lixiang1201_2323_optimize-rag-performance` 的優化，重用 `main.py` 中建立的單一 `ollama.Client` 實例，避免重複連線開銷。

## 🚀 未來工作 (Future Work)

現在我們已經成功整合了混合檢索與 LLM 重排序機制，大幅提升了檢索的精準度。接下來，我們將專注於以下更進階的優化和探索：

#### 1. RAG 流程強化 (RAG Workflow Enhancements)
- **進階重排序模型 (Advanced Reranking Models):** 探索並整合更專業的重排序模型 (例如 BGE Reranker)，以進一步提升排序精準度並可能降低延遲。
- **查詢改寫與擴展 (Query Rewriting & Expansion):**
  - **Multi-Query Generation:** 利用 LLM 從單一查詢生成多個視角的問題，擴展檢索範圍。
  - **HyDE (Hypothetical Document Embeddings):** 生成假設性文件並用於檢索，捕捉查詢的語意意圖。
  - **Decomposition:** 將複雜查詢分解成可獨立回答的子問題，提升RAG處理複雜問題的能力。
  - **Step-Back Prompting:** 讓模型從具體問題回溯到更高層次的抽象概念，幫助檢索更相關的背景資訊。

#### 2. 文件處理與管理 (Document Processing & Management)
- **智能切分策略 (Intelligent Chunking Strategies):** 深入研究並實作如 `RecursiveCharacterTextSplitter` 等基於內容結構的智能切分方法，以確保資訊單元完整性，並系統性地測試 `chunk_size` 和 `chunk_overlap` 對 RAG 表現的影響。
- **知識圖譜整合 (Knowledge Graph Integration):** 探索從文件內容構建知識圖譜的可能性，並利用圖譜檢索來處理複雜、多跳的查詢。

#### 3. 生成模型與互動 (Generation & Interaction)
- **上下文視窗優化 (Context Window Optimization):** 精煉傳遞給生成模型的上下文，在有限的 Token 視窗內提供最關鍵且無冗餘的資訊，可能涉及摘要、資訊壓縮等技術。
- **動態 Prompt Engineering (Dynamic Prompt Engineering):** 開發能夠根據查詢類型、檢索結果等動態調整提示詞的機制，以引導生成模型產生更精確、符合要求的答案。
- **答案格式規範與驗證 (Answer Formatting & Validation):** 強化生成答案的格式控制與內容驗證，確保輸出的一致性與品質。

#### 4. 系統級優化與評估 (System-Level Optimization & Evaluation)
- **實時效能監控 (Real-time Performance Monitoring):** 建立 RAG 系統各環節的實時監控，識別瓶頸並進行優化。
- **持續評估框架 (Continuous Evaluation Framework):** 建立自動化的評估流程，能夠快速反饋不同優化策略的效果。
- **模型微調探索 (Model Fine-tuning Exploration):** 針對特定任務和資料集，微調嵌入模型、重排序模型或生成模型，以獲得更高的專案效能。
