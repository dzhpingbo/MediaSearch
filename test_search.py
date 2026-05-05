"""测试检索功能"""
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

from search import MediaSearchEngine

engine = MediaSearchEngine(r"E:\dzhwork\pictureVedioIndex")
print("=== 索引统计 ===")
print(engine.stats())
print()

for query in ["孩子", "截图", "文档", "照片"]:
    print(f"=== 搜索: {query} ===")
    results = engine.search(query, top_k=3)
    if not results:
        print("  (无结果)")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['filename']}")
    print()
