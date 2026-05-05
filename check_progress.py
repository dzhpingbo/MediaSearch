"""检查索引构建进度"""
import os
import json
import time

INDEX_DIR = r"E:\dzhwork\pictureVedioIndex"
META_PATH = os.path.join(INDEX_DIR, "metadata.json")
INDEX_PATH = os.path.join(INDEX_DIR, "vectors.faiss")

print("=" * 60)
print("MediaSearch 索引进度检查")
print("=" * 60)

if not os.path.exists(META_PATH):
    print("\n[X] 索引尚未开始或 metadata.json 未生成")
    print(f"    路径: {META_PATH}")
else:
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    count = len(metadata)
    
    # 获取文件修改时间
    mtime = os.path.getmtime(META_PATH)
    mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
    
    print(f"\n[OK] 当前已索引: {count} 个文件")
    print(f"[OK] 最近更新: {mtime_str}")
    
    if count > 0:
        print("\n最近索引的文件:")
        for item in metadata[-3:]:
            print(f"  - {item['filename']} ({item['type']})")

if os.path.exists(INDEX_PATH):
    size_mb = os.path.getsize(INDEX_PATH) / 1024 / 1024
    print(f"\n[OK] FAISS 索引文件: {size_mb:.2f} MB")

print("\n" + "=" * 60)
print("提示: 运行 'python build_index.py' 可继续构建索引")
print("=" * 60)
