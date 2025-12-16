#!/bin/bash
set -e

echo "[Build] Installing dependencies..."
pip install -r requirements.txt -q

# 2. 還原模型 (Execution Phase 斷網，必須從碎片還原)
echo "[Setup] Checking for bundled models..."
if [ -d "model_chunks" ]; then
    echo "[Setup] Decompressing models from chunks..."
    python decompress_models.py
fi

# 3. 雙重保險：如果解壓縮失敗，才嘗試下載 (但在斷網環境會失敗，僅供本地測試用)
echo "[Setup] Verifying Embedding Model..."
if [ ! -d "./local_bge_m3" ] || [ -z "$(ls -A ./local_bge_m3)" ]; then
    echo "⚠️ Local model not found after decompression. Attempting online download..."
    python ./download_model.py
else
    echo "✅ Embedding Model found locally."
fi

# 4. 啟動 Ollama
if ! pgrep -x "ollama" > /dev/null
then
    echo "Starting Ollama..."
    ollama serve &
    
    count=0
    while ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
        echo "Waiting for Ollama... ($count/30)"
        sleep 1
        count=$((count+1))
        if [ $count -ge 30 ]; then
            echo "Error: Ollama failed to start."
            exit 1
        fi
    done
    echo "Ollama is ready!"
fi

# 5. 下載 Qwen (斷網時這裡會失敗，除非伺服器有快取)
# 注意：如果比賽環境完全斷網且沒有預載 Qwen，你可能也需要打包 Qwen 的 GGUF 檔
if ! ollama list | grep -q "qwen2.5:3b"; then
    echo "Pulling Qwen model..."
    ollama pull qwen2.5:3b || echo "⚠️ Failed to pull Qwen. Hopefully it's pre-loaded or we have a backup."
fi

# 6. 執行主程式
log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "========================================"
    echo "$timestamp - $1"
    echo "========================================"
}

run_results() {
    local language=$1
    log "[INFO] Running inference for language: ${language}"
    
    python ./My_RAG/main.py \
        --query_path ./data/queries_show/queries_${language}.jsonl \
        --docs_path ./data/dragonball_docs.jsonl \
        --language ${language} \
        --output ./predictions/predictions_${language}.jsonl

    log "[INFO] Checking output format..."
    python ./check_output_format.py \
        --query_file ./data/queries_show/queries_${language}.jsonl \
        --processed_file ./predictions/predictions_${language}.jsonl
}

mkdir -p ./predictions
run_results "en"
run_results "zh"

log "[INFO] All tasks completed."