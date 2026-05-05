"""测试 deepface 人脸识别（使用 ONNX 后端，避免 TensorFlow）"""
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

# 禁用 TensorFlow 导入
os.environ["DEEPFACE_BACKEND"] = "onnx"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("=" * 60)
print("测试 deepface 人脸识别 (ONNX 后端)")
print("=" * 60)

try:
    from deepface import DeepFace
    import numpy as np
    
    # 测试图片
    test_img = r"C:\Users\Administrator\Pictures\微信图片_20250713221151.jpg"
    
    print(f"\n[1] 检测人脸并提取特征 (使用 ArcFace-ONNX)...")
    # 使用 ArcFace 模型 (支持 ONNX，不需要 TensorFlow)
    result = DeepFace.represent(
        img_path=test_img,
        model_name="ArcFace",
        detector_backend="opencv",  # 使用 OpenCV 检测人脸，不依赖 TensorFlow
        enforce_detection=False,
        silent=True
    )
    
    if result and len(result) > 0:
        embedding = result[0]["embedding"]
        print(f"[OK] 特征向量维度: {len(embedding)}")
        print(f"     前5维: {[f'{x:.3f}' for x in embedding[:5]]}")
    
    print(f"\n[2] 人脸分析 (年龄/性别/情绪)...")
    analysis = DeepFace.analyze(
        img_path=test_img,
        actions=['age', 'gender', 'emotion'],
        detector_backend="opencv",
        enforce_detection=False,
        silent=True
    )
    
    if analysis and len(analysis) > 0:
        print(f"[OK] 分析结果:")
        for key, val in analysis[0].items():
            if key not in ['embedding', 'facial_area', 'face_confidence']:
                print(f"    {key}: {val}")
    
    print("\n" + "=" * 60)
    print("deepface 测试完成！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()
