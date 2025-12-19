# Conda 環境設定與執行指南

本文件說明如何使用 Conda 設定環境並直接在本地端執行 RAG 應用程式，不透過 Docker。

## 1. 事前準備

在開始之前，請確保您已經在您的系統上安裝了以下軟體：

- **Miniconda 或 Anaconda**: [Conda 安裝指南](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- **Ollama**: [Ollama 官網](https://ollama.com/)

## 2. 環境設定

### 步驟 1: 啟動 Ollama 並下載模型

安裝完 Ollama 後，請在一個**獨立的終端機**中執行以下指令來啟動 Ollama 服務：

```bash
ollama serve
```

接著，打開**另一個新的終端機**，下載本專案所需的兩個模型：

```bash
ollama pull gemma:2b
ollama pull granite4:3b
```

請確保這兩個模型都已成功下載，並且 `ollama serve` 的終端機保持開啟狀態。

### 步驟 2: 建立 Conda 環境並安裝套件

1.  **建立 Conda 環境**:
    開啟終端機，執行以下指令建立一個名為 `wsm_final` 的 Python 3.10 環境。

    ```bash
    conda create -n wsm_final python=3.10 -y
    ```

2.  **啟用 Conda 環境**:
    ```bash
    conda activate wsm_final
    ```
    您應該會看到終端機提示符號前面出現 `(wsm_final)`。

3.  **安裝 Python 套件**:
    在此環境中，使用 `pip` 安裝 `requirements.txt` 中列出的所有套件。

    ```bash
    pip install -r requirements.txt
    ```

## 3. 執行應用程式

完成所有設定後，請確保您仍處於 `wsm_final` Conda 環境中。

由於您將在 WSL/Linux 環境中執行，請先給予 `run.sh` 執行權限：

```bash
chmod +x run.sh
```

然後，直接執行腳本即可：

```bash
./run.sh
```

這個腳本會自動設定 `OLLAMA_HOST` 環境變數，並執行 Python 主程式，將結果輸出到 `predictions` 和 `results` 資料夾中。

```