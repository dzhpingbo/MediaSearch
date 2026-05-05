#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速测试搜索功能是否正常"""
import os, sys
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"
# 强制 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from search import MediaSearchEngine

INDEX_DIR = r"E:\dzhwork\pictureVedioIndex"

print("=" * 60)
print("MediaSearch 搜索功能测试")
print("=" * 60)

engine = MediaSearchEngine(INDEX_DIR)
stats = engine.stats()
print(f"\n[索引统计]")
print(f"  总文件数：{stats['total']}")
print(f"  图片：{stats['images']}")
print(f"  视频：{stats['videos']}")
print(f"  文档：{stats.get('documents', 0)}")

# 测试不同关键词
test_queries = [
    ("孩子", None),
    ("证件", None),
    ("截图", {"image"}),
    ("视频录制", {"video"}),
]

for query, file_types in test_queries:
    type_str = str(file_types) if file_types else "全部"
    print(f"\n[搜索] \"{query}\" (类型:{type_str}):")
    results = engine.search(query, top_k=3, file_types=file_types, min_score=0.1)
    if results:
        for r in results:
            type_tag = r.get("type", "?")
            print(f"  [{r['score']:.3f}] [{type_tag}] {r['filename']}")
    else:
        print("  (无结果)")

print("\n[OK] 搜索功能正常！")
