#!/bin/bash
set -e

# 1. 安裝套件
# 加上 -q (quiet) 可以減少 log 雜訊，讓你看 log 更清楚
echo "[Build] Installing dependencies..."
pip install -r requirements.txt -q

echo "[Build] Checking Embedding Model..."
if [ -f "./My_RAG/download_model.py" ]; then
    python ./My_RAG/download_model.py
elif [ -f "./download_model.py" ]; then
    python ./download_model.py
else
    echo "Error: download_model.py not found!"
    exit 1
fi

if ! pgrep -x "ollama" > /dev/null
then
    echo "Starting Ollama..."
    ollama serve &
    
    # [專家優化] 使用迴圈檢查 Ollama 是否真的活著，而不是傻傻等 5 秒
    count=0
    while ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
        echo "Waiting for Ollama to launch... ($count/30)"
        sleep 1
        count=$((count+1))
        if [ $count -ge 30 ]; then
            echo "Error: Ollama failed to start timeout."
            exit 1
        fi
    done
    echo "Ollama is ready!"
else
    echo "Ollama is already running."
fi

# 4. 下載 Qwen 模型 (斷網保護關鍵！)
if ! ollama list | grep -q "qwen2.5:3b"; then
    echo "Model qwen2.5:3b not found. Pulling..."
    ollama pull qwen2.5:3b || echo "Warning: Failed to pull model (Offline mode?)"
else
    echo "Model qwen2.5:3b found locally. Skipping download."
fi

# 5. 定義 Log 函數
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="$timestamp - $1"
    local len=${#message}
    local border=$(printf '=%.0s' $(seq 1 $len))
    
    echo "$border"
    echo "$message"
    echo "$border"
}

# 6. 定義執行函數
run_results() {
    local language=$1

    log "[INFO] Running inference for language: ${language}"
    
    # 執行主程式
    python ./My_RAG/main.py \
        --query_path ./data/queries_show/queries_${language}.jsonl \
        --docs_path ./data/dragonball_docs.jsonl \
        --language ${language} \
        --output ./predictions/predictions_${language}.jsonl

    log "[INFO] Checking output format for language: ${language}"
    python ./check_output_format.py \
        --query_file ./data/queries_show/queries_${language}.jsonl \
        --processed_file ./predictions/predictions_${language}.jsonl

    if [ $? -eq 0 ]; then
        echo "Format check passed for ${language}."
    else
        echo "Format check FAILED for ${language}!"
    fi
}

mkdir -p ./predictions

run_results "en"
run_results "zh"

log "[INFO] All inference tasks completed."