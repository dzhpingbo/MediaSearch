#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试文档处理功能"""
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

print("=" * 60)
print("测试文档处理功能")
print("=" * 60)

# 测试1：提取 TXT 文本
print("\n[测试1] TXT 文本提取")
from core_encoder import extract_txt_text
test_txt = r"C:\MediaSearch\test.txt"
if os.path.exists(test_txt):
    text = extract_txt_text(test_txt)
    print(f"  提取内容: {text[:200]}...")
else:
    print(f"  文件不存在: {test_txt}")

# 测试2：编码文档
print("\n[测试2] 文档编码")
from core_encoder import encode_document
# 用一个已知存在的文件测试
test_file = r"C:\MediaSearch\test_search.py"
if os.path.exists(test_file):
    vec = encode_document(test_file)
    if vec is not None:
        print(f"  向量维度: {vec.shape}")
        print(f"  向量范数: {vec.norm()}")
    else:
        print("  编码失败")
else:
    print(f"  文件不存在: {test_file}")

# 测试3：扫描文件（包括文档）
print("\n[测试3] 文件扫描（包括文档）")
from build_index import scan_files
test_dir = r"C:\MediaSearch"
files = scan_files([test_dir], max_files=20)
print(f"  扫描到 {len(files)} 个文件")
for f in files[:5]:
    print(f"    - {f['type']:10s} {f['path']}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
