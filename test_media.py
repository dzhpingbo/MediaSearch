# MediaSearch 测试脚本
import sys
import os
sys.path.insert(0, r"C:\MediaSearch")

# 禁用代理
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import numpy as np
from PIL import Image
from core_encoder import encode_image, encode_text, describe_image

# 测试照片列表（选取不同类型）
TEST_IMAGES = [
    r"C:\Users\Administrator\Pictures\微信图片_20250713221151.jpg",  # 微信照片1
    r"C:\Users\Administrator\Pictures\微信图片_20260311003426_322_4.jpg",  # 微信照片2
    r"C:\Users\Administrator\Pictures\印章.jpg",  # 印章
    r"C:\Users\Administrator\Pictures\专利申请.png",  # 专利截图
    r"C:\Users\Administrator\Pictures\dzh1.jpg",  # 照片
    r"C:\Users\Administrator\Pictures\images.jpg",  # 素材图
]

print("=" * 60)
print("MediaSearch Chinese-CLIP 效果测试")
print("=" * 60)

# 1. 测试文本编码
print("\n[1] 文本编码测试")
test_queries = [
    "一只猫",
    "风景照片",
    "文档截图",
    "印章",
    "人物照片",
]
print("\n生成各查询向量...")
query_vecs = {}
for q in test_queries:
    query_vecs[q] = encode_text(q)
    print(f"  - '{q}' OK")

# 2. 对测试图片编码并生成描述
print("\n[2] 图片编码 + 描述生成")
image_results = []
for img_path in TEST_IMAGES:
    name = os.path.basename(img_path)
    print(f"\n处理: {name}")
    try:
        vec = encode_image(img_path)
        desc = describe_image(img_path)
        image_results.append({
            "path": img_path,
            "name": name,
            "vec": vec,
            "desc": desc
        })
        print(f"  描述: {desc[:80]}...")
    except Exception as e:
        print(f"  失败: {e}")

# 3. 语义检索测试
print("\n[3] 语义检索测试")
for query, qvec in query_vecs.items():
    print(f"\n查询: '{query}'")
    scores = []
    for img in image_results:
        sim = np.dot(qvec, img["vec"])
        scores.append((sim, img["name"], img["desc"][:40]))
    scores.sort(reverse=True)
    print("  Top 3 结果:")
    for i, (score, name, desc) in enumerate(scores[:3], 1):
        print(f"    {i}. [{score:.3f}] {name} - {desc}")

# 4. 以图搜图测试
print("\n[4] 以图搜图测试（用第一张图搜相似图）")
if image_results:
    ref_vec = image_results[0]["vec"]
    ref_name = image_results[0]["name"]
    print(f"参考图: {ref_name}")
    scores = []
    for img in image_results:
        if img["name"] != ref_name:
            sim = np.dot(ref_vec, img["vec"])
            scores.append((sim, img["name"]))
    scores.sort(reverse=True)
    print("  相似图片 Top 3:")
    for i, (score, name) in enumerate(scores[:3], 1):
        print(f"    {i}. [{score:.3f}] {name}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
