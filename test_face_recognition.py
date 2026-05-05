"""测试 deepface 人脸识别"""
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

print("=" * 60)
print("测试 deepface 人脸识别")
print("=" * 60)

try:
    from deepface import DeepFace
    import numpy as np
    
    # 测试图片
    test_img = r"C:\Users\Administrator\Pictures\微信图片_20250713221151.jpg"
    
    print(f"\n[1] 检测人脸: {test_img}")
    # 只检测人脸，不提取特征
    from deepface.commons import functions
    faces = DeepFace.extract_faces(img_path=test_img, detector_backend="opencv", enforce_detection=False)
    
    print(f"[OK] 检测到 {len(faces)} 个人脸")
    
    if len(faces) > 0:
        print("\n[2] 提取人脸特征...")
        embedding = DeepFace.represent(img_path=test_img, model_name="Facenet", enforce_detection=False)
        print(f"[OK] 特征向量维度: {len(embedding[0]['embedding'])}")
        
        print("\n[3] 人脸分析 (年龄/性别/情绪)...")
        analysis = DeepFace.analyze(img_path=test_img, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        print(f"[OK] 分析结果:")
        for key, val in analysis[0].items():
            if key not in ['embedding', 'facial_area']:
                print(f"    {key}: {val}")
    
    print("\n" + "=" * 60)
    print("deepface 测试完成！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()
