"""直接用本地路径测试 BLIP 加载"""
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 本地快照路径
LOCAL_PATH = r"C:\Users\Administrator\.cache\huggingface\hub\models--Salesforce--blip-image-captioning-large\snapshots\353689b859fcf0523410b1806dace5fb46ecdf41"

print("测试 BLIP 本地加载...")
print(f"路径: {LOCAL_PATH}")

try:
    print("\n[1] 加载 Processor...")
    processor = BlipProcessor.from_pretrained(LOCAL_PATH, use_fast=False)
    print("[OK] Processor 加载成功")
    
    print("\n[2] 加载 Model (float16)...")
    model = BlipForConditionalGeneration.from_pretrained(
        LOCAL_PATH,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    print("[OK] Model 加载成功")
    
    print("\n[3] 测试图片描述...")
    # 创建测试图片
    test_img = Image.new("RGB", (512, 512), color=(120, 180, 220))
    inputs = processor(test_img, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"[OK] 测试描述: {caption}")
    
    print("\n" + "=" * 60)
    print("BLIP 本地加载成功！可以修复 core_encoder.py 了")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()
