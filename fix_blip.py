"""
修复 BLIP 描述模型下载问题
使用 hf-mirror.com 镜像正确下载 BLIP 模型文件
"""
import os
import sys

# 在所有 import 之前禁用代理
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from huggingface_hub import snapshot_download

print("=" * 60)
print("修复 BLIP 模型下载")
print("=" * 60)

# 方法1：用 snapshot_download 先下载完整模型到本地缓存
print("\n[1] 下载 BLIP 模型文件到本地缓存...")
try:
    cache_dir = snapshot_download(
        repo_id="Salesforce/blip-image-captioning-large",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        resume_download=True,
        max_workers=4,
    )
    print(f"[OK] 模型已下载到: {cache_dir}")
except Exception as e:
    print(f"[错误] 下载失败: {e}")
    print("\n尝试方法2：手动指定本地路径...")
    cache_dir = None

# 方法2：如果下载成功，测试加载
if cache_dir:
    print("\n[2] 测试加载 BLIP 模型...")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
            torch_dtype=torch.float16,
        ).to("cuda")
        model.eval()
        print("[OK] BLIP 模型加载成功！")
        
        # 测试一张图片
        print("\n[3] 测试图片描述生成...")
        from PIL import Image
        test_img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        inputs = processor(test_img, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)
        print(f"[OK] 测试描述: {caption}")
        
    except Exception as e:
        print(f"[错误] 加载失败: {e}")

print("\n" + "=" * 60)
print("修复完成")
print("=" * 60)
