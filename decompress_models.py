import os
import zipfile
import shutil
import glob

SOURCE_DIR = "model_chunks"
TARGETS = [
    "local_bge_m3",
    "local_bge_reranker"
]

def merge_files(base_name):
    """合併碎片檔案"""
    parts = sorted(glob.glob(f"{base_name}.part*"))
    if not parts:
        return False
    
    print(f"🔗 Merging {base_name} from {len(parts)} parts...")
    with open(base_name, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    return True

def decompress():
    if not os.path.exists(SOURCE_DIR):
        print("⚠️ No model chunks found. Skipping decompression.")
        return

    for target in TARGETS:
        # 如果目標資料夾已經存在且不為空，跳過
        if os.path.exists(target) and os.listdir(target):
            print(f"✅ {target} already exists. Skipping.")
            continue

        zip_path = os.path.join(SOURCE_DIR, f"{target}.zip")
        
        # 1. 合併
        if merge_files(zip_path):
            # 2. 解壓縮
            print(f"📂 Unzipping {zip_path} to {target}...")
            # 確保目標路徑是乾淨的
            if os.path.exists(target):
                shutil.rmtree(target)
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(target)
            
            # 清理暫存的合併檔 (可選)
            os.remove(zip_path)
            print(f"✅ Restored {target}")
        else:
            print(f"⚠️ Could not find chunks for {target}")

if __name__ == "__main__":
    decompress()