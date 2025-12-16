import os
import zipfile
import shutil

# 設定要打包的資料夾與目標路徑
TARGETS = [
    "local_bge_m3",
    "local_bge_reranker"
]
OUTPUT_DIR = "model_chunks"
CHUNK_SIZE = 90 * 1024 * 1024  # 90MB (預留緩衝，避開GitHub 100MB限制)

def split_file(file_path, chunk_size):
    """將大檔案切割成小塊"""
    with open(file_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_name = f"{file_path}.part{chunk_num:03d}"
            with open(chunk_name, 'wb') as chunk_f:
                chunk_f.write(chunk)
            print(f"  -> Created chunk: {chunk_name}")
            chunk_num += 1
    # 移除原始大檔
    os.remove(file_path)

def compress_and_split():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for target in TARGETS:
        if not os.path.exists(target):
            print(f"⚠️ Warning: {target} not found. Skipping.")
            continue
        
        print(f"📦 Compressing {target}...")
        zip_name = os.path.join(OUTPUT_DIR, f"{target}.zip")
        
        # 1. 壓縮
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(target):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=target)
                    zipf.write(file_path, arcname)
        
        # 2. 切割
        print(f"✂️ Splitting {zip_name}...")
        split_file(zip_name, CHUNK_SIZE)

    print("\n✅ Compression and splitting complete! Please push the 'model_chunks' folder to git.")

if __name__ == "__main__":
    compress_and_split()