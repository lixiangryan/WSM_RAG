FROM ollama/ollama

# [新增] 安裝 curl 工具
# ollama 映像檔是基於 Ubuntu 的，所以我們用 apt-get
RUN apt-get update && apt-get install -y curl

# 複製我們剛剛建立的 entrypoint 腳本
COPY entrypoint.sh /entrypoint.sh

# [新增] 移除 Windows 的換行符號 (CRLF)
RUN sed -i 's/\r$//' /entrypoint.sh

# 讓腳本可以被執行
RUN chmod +x /entrypoint.sh

# 告訴 container 啟動時要執行這個腳本
ENTRYPOINT ["/entrypoint.sh"]