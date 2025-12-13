#!/bin/sh

# 1. 等待 Ollama 伺服器本身啟動
echo "等待 Ollama 伺服器啟動..."
attempts=0
max_attempts=60 # 等待最多 120 秒 (increased from 60s)
until curl --output /dev/null --silent --fail http://ollama:11434/api/tags; do
    if [ $attempts -ge $max_attempts ]; then
        echo "錯誤: Ollama 伺服器未啟動。"
        echo "嘗試連接 http://ollama:11434 失敗"
        echo "正在嘗試診斷網路連接..."
        ping -c 3 ollama || echo "無法 ping ollama 主機"
        exit 1
    fi
    printf '.'
    attempts=$((attempts+1))
    sleep 2
done
echo "\nOllama 伺服器已準備就緒。"

# 2. 定義需要檢查的模型列表 (從環境變數讀取)
MODELS_TO_CHECK="$GENERATOR_MODEL $JUDGE_MODEL"

# 3. 遍歷定義的每個必要模型，並等待它們下載完成
echo "正在檢查必要的模型: $MODELS_TO_CHECK"

for model_name in $MODELS_TO_CHECK; do
    # 如果模型名稱為空，則跳過
    if [ -z "$model_name" ]; then
        continue
    fi
    
    echo "正在等待模型 '$model_name' 下載完成..."
    model_attempts=0
    max_model_attempts=300 # 為每個模型提供最多 10 分鐘的下載時間

    # 使用 jq 的 -e 選項，如果找到匹配項，它會返回 0 (成功)
    until curl -s http://ollama:11434/api/tags | jq -e ".models[] | select(.name == \"$model_name\")" > /dev/null 2>&1; do
        if [ $model_attempts -ge $max_model_attempts ]; then
            echo "錯誤: 等待模型 '$model_name' 超時。"
            exit 1
        fi
        printf '.'
        model_attempts=$((model_attempts+1))
        sleep 2
    done
    echo "\n模型 '$model_name' 已準備就緒。"
done

# 4. 所有模型都準備好後，才執行主程式
echo "\n所有必要模型均已準備就緒，開始執行主程式..."
exec ./run.sh