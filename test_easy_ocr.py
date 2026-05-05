"""测试 EasyOCR 中文识别效果"""
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

print("=" * 60)
print("测试 EasyOCR 中文识别")
print("=" * 60)

try:
    print("\n[1] 初始化 EasyOCR (中文+英文)...")
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True, verbose=False)
    print("[OK] EasyOCR 初始化成功")
    
    # 测试一张图片
    test_img = r"C:\Users\Administrator\Pictures\微信图片_20250713221151.jpg"
    print(f"\n[2] OCR 识别: {test_img}")
    result = reader.readtext(test_img)
    
    print(f"\n识别到 {len(result)} 个文本区域:")
    texts = []
    for (bbox, text, conf) in result:
        texts.append(text)
        print(f"  - '{text}' (置信度: {conf:.2f})")
    
    print(f"\n[OK] 共识别 {len(texts)} 段文本")
    print(f"合并文本: {' | '.join(texts)}")
    
except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()
