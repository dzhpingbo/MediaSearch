"""手动下载 EasyOCR 模型（使用 GitHub 镜像或直连）"""
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import urllib.request
import sys

print("=" * 60)
print("手动下载 EasyOCR 模型")
print("=" * 60)

# EasyOCR 模型下载 URL
MODELS = {
    "craft_mlt_25k.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.pth",
    "zh_sim_g2.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.4/zh_sim_g2.pth",
    "english_g2.pth": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.pth",
}

# 模型存放目录
MODEL_DIR = os.path.expanduser("~/.EasyOCR/")
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"模型存放目录: {MODEL_DIR}")

# 尝试多个镜像源
MIRRORS = [
    "",  # 原始 GitHub
    "https://ghproxy.com/",  # GitHub 代理
    "https://mirror.ghproxy.com/",  # 另一个代理
]

for name, url in MODELS.items():
    save_path = os.path.join(MODEL_DIR, name)
    if os.path.exists(save_path):
        print(f"\n[跳过] {name} 已存在")
        continue
    
    print(f"\n[下载] {name} ...")
    success = False
    for mirror in MIRRORS:
        try:
            full_url = mirror + url
            print(f"  尝试: {mirror if mirror else '直连'}")
            urllib.request.urlretrieve(full_url, save_path)
            print(f"  [OK] 下载成功: {save_path}")
            success = True
            break
        except Exception as e:
            print(f"  [失败] {e}")
            continue
    
    if not success:
        print(f"[错误] 所有镜像都失败，请手动下载: {url}")
        print(f"      然后放到: {save_path}")
        sys.exit(1)

print("\n" + "=" * 60)
print("所有模型下载完成！")
print("=" * 60)
