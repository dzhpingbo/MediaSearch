"""测试 PaddleOCR 安装和中文OCR效果"""
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

print("=" * 60)
print("测试 PaddleOCR 安装")
print("=" * 60)

try:
    print("\n[1] 安装 paddlepaddle-gpu...")
    import subprocess, sys
    # 先尝试安装 paddlepaddle-gpu (CUDA 11.7 对应版本)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "paddlepaddle-gpu==2.6.0", "-i", "https://mirror.baidu.com/pypi/simple"],
        capture_output=True, text=True
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print("尝试 CPU 版本...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "paddlepaddle==2.6.0", "-i", "https://mirror.baidu.com/pypi/simple"],
            capture_output=True, text=True
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    
    print("\n[2] 安装 paddleocr...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "paddleocr==2.7.0.3", "-i", "https://mirror.baidu.com/pypi/simple"],
        capture_output=True, text=True
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    
    print("\n[3] 测试 OCR...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True, show_log=False)
    print("[OK] PaddleOCR 初始化成功")
    
    # 测试一张图片
    test_img = r"C:\Users\Administrator\Pictures\微信图片_20250713221151.jpg"
    print(f"\n[4] OCR 识别测试图片: {test_img}")
    result = ocr.ocr(test_img, cls=True)
    
    print("\n识别结果:")
    texts = []
    for line in result[0]:
        text = line[1][0]
        conf = line[1][1]
        texts.append(text)
        print(f"  - {text} (置信度: {conf:.2f})")
    
    print(f"\n[OK] 共识别 {len(texts)} 个文本区域")
    
except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()
