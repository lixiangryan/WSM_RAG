#!/bin/sh

# [新增] 檢查服務是否在運行 (延續之前的 curl 檢查邏輯)
echo "等待 Ollama 伺服器啟動..."
attempts=0
max_attempts=30
# --- [修正語法] ---
# 移除 $()，直接執行 curl，讓其返回碼控制 until 迴圈
until curl --output /dev/null --silent --fail http://ollama:11434/api/tags; do
# --------------------
    if [ $attempts -ge $max_attempts ]; then
        echo "錯誤: Ollama 伺服器未啟動。"
        exit 1
    fi
    printf '.'
    attempts=$((attempts+1))
    sleep 2
done
echo "\nOllama 伺服器已準備就緒。"

# --- [新增這裡] ---
# 強制載入模型到 GPU VRAM (觸發一次簡單的生成請求)
echo "強制載入 gemma:2b 模型到 VRAM (可能耗時 10-30 秒)..."
# 確保這個 curl 指令能夠成功發出
curl -s -X POST http://ollama:11434/api/generate -d '{
  "model": "gemma:2b",
  "prompt": "ping",
  "stream": false
}' > /dev/null 2>&1 & # 丟到背景執行

# 等待模型載入完成 (給予足夠時間)
sleep 40 
# --------------------

echo "開始執行 RAG 預測和評估..."
exec ./run.sh