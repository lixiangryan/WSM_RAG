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

如果你沒有修改任何程式碼，只想重新跑一次評估，你只需要執行：

```bash
docker-compose up
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

## lixiang_1114當前任務
紀錄每次結果並加上時間戳記

## 結果在這裡
執行 `run.sh` 腳本後，結果將會儲存在專案根目錄下的 `result` 和 `predictions` 資料夾中。這兩個資料夾都會包含以時間戳記命名的子資料夾，例如 `result/20251114_074809/` 和 `predictions/20251114_074809/`。

1.  **`result` 資料夾：**
    *   包含最終的「評估分數」檔案。
    *   例如：`result/YYYYMMDD_HHMMSS/score_en.jsonl` 和 `result/YYYYMMDD_HHMMSS/score_zh.jsonl`。
    *   這些檔案包含了 RAG 評估的最終分數。

2.  **`predictions` 資料夾：**
    *   包含中間的「RAG 預測」檔案。
    *   例如：`predictions/YYYYMMDD_HHMMSS/predictions_en.jsonl` 和 `predictions/YYYYMMDD_HHMMSS/predictions_zh.jsonl`。
    *   這些是 `rageval` 讀取並用來計算分數的原始答案。

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

## 🚀 未來工作 (Future Work)

### 主要目標：修正 RAG 檢索流程

當前的評估結果不佳，根本原因並非模型（裁判或生成模型）能力不足，而是檢索階段無法從 `dragonball_docs.jsonl` 中找到相關文件。

### 代辦事項 (To-Do)

- [ ] **偵錯 `My_RAG/main.py` 中的檢索演算法：**
  - [ ] **分析 Chunking 策略：** 檢查文件切分是否合理，有沒有遺失關鍵資訊。
  - [ ] **驗證 Embedding 模型：** 確認 Embedding 的效果是否能有效表達文本語意。
  - [ ] **檢視 Similarity Search：** 檢查相似度搜索的邏輯，確認是否能正確匹配查詢與文件。
- [ ] **暫緩更換模型：** 在檢索問題解決前，無需更換 `gemma:2b` 或 `granite4:3b` 模型。
