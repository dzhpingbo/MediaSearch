# ── 必须在所有 import 之前禁用代理，否则 httpx 会缓存代理设置 ──
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

"""
core_encoder.py - 图片/视频特征提取核心（Chinese-CLIP 版）
使用 Chinese-CLIP 提取视觉向量 + BLIP 生成描述
支持中文自然语言检索

运行环境要求：
  conda activate aimodel
  Python 3.11.4 | torch 2.0.1+cu117 | GTX 1080 Ti
"""
import json
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Optional

# ── torch 版本守卫（防止在错误的 torch 版本下运行，cu117 专用）──────
_tv = torch.__version__
if not _tv.startswith("2.0") or "cu117" not in _tv:
    raise EnvironmentError(
        f"[Encoder] torch 版本不对！当前: {_tv}\n"
        f"需要 2.0.1+cu117，请运行：\n"
        f"  pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 "
        f"--index-url https://download.pytorch.org/whl/cu117"
    )

# ── 全局设备 ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Encoder] 使用设备: {DEVICE}  (torch {torch.__version__})")

# ── Chinese-CLIP 模型加载 ─────────────────────────────────────
# 可用模型: ViT-B/16(快) ViT-L/14(推荐) ViT-L/14-336(精细) ViT-H/14(最强)
# 中文理解原生支持，不需要中英文映射模板
_CN_CLIP_MODEL = "ViT-L-14"   # ← 如需更高质量改为 "ViT-L-14-336"

_cn_model = None
_cn_preprocess = None

def get_cn_clip_model():
    """加载 Chinese-CLIP 模型（首次运行自动从 hf-mirror 下载）"""
    global _cn_model, _cn_preprocess
    if _cn_model is None:
        # 禁用代理，避免 SOCKS 代理导致 httpx 报错
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from cn_clip.clip import load_from_name
        print(f"[Encoder] 加载 Chinese-CLIP 模型 ({_CN_CLIP_MODEL})...")
        _cn_model, _cn_preprocess = load_from_name(
            _CN_CLIP_MODEL,
            device=DEVICE,
            download_root=os.path.expanduser("~/.cache/chinese_clip")
        )
        _cn_model.eval()
        # 打印向量维度，用于确认 FAISS 索引维度
        with torch.no_grad():
            dummy = _cn_preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(DEVICE)
            feat = _cn_model.encode_image(dummy)
            DIM = feat.shape[-1]
        print(f"[Encoder] Chinese-CLIP 加载完成，向量维度: {DIM}")
        global VEC_DIM
        VEC_DIM = DIM
    return _cn_model, _cn_preprocess


# ── BLIP 图像描述模型（可选，生成图片描述） ────────────────────
_blip_processor = None
_blip_model = None
_BLIP_LOCAL_PATH = None  # 动态获取本地路径

def _get_blip_local_path():
    """获取 BLIP 模型的本地缓存路径（自动下载 if needed）"""
    global _BLIP_LOCAL_PATH
    if _BLIP_LOCAL_PATH is not None:
        return _BLIP_LOCAL_PATH
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print("[Encoder] 检查 BLIP 模型缓存...")
    cache_dir = snapshot_download(
        repo_id="Salesforce/blip-image-captioning-large",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
        resume_download=True,
    )
    _BLIP_LOCAL_PATH = cache_dir
    print(f"[Encoder] BLIP 本地路径: {cache_dir}")
    return cache_dir

def get_blip_model():
    """加载 BLIP 模型生成图片描述（英文，可后续翻译）"""
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("[Encoder] 加载 BLIP 描述模型...")
        local_path = _get_blip_local_path()
        _blip_processor = BlipProcessor.from_pretrained(local_path, use_fast=False)
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            local_path, torch_dtype=torch.float16
        ).to(DEVICE)
        _blip_model.eval()
        print("[Encoder] BLIP 加载完成")
    return _blip_processor, _blip_model


# ── 图片编码 ─────────────────────────────────────────────────
VEC_DIM = None  # 在 get_cn_clip_model() 中动态确定

def encode_image(image_path: str) -> Optional[np.ndarray]:
    """提取图片的 Chinese-CLIP 向量"""
    try:
        model, preprocess = get_cn_clip_model()
        img = Image.open(image_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)[0]
    except Exception as e:
        print(f"[Encoder] 图片编码失败 {image_path}: {e}")
        return None


def describe_image(image_path: str) -> str:
    """用 BLIP 生成图片描述"""
    try:
        processor, model = get_blip_model()
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)
        inputs = processor(img, return_tensors="pt").to(DEVICE, torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"[Encoder] 图片描述失败 {image_path}: {e}")
        return ""


# ── 视频处理（抽帧取平均向量） ─────────────────────────────────
def extract_video_frames(video_path: str, max_frames: int = 8) -> list:
    """从视频均匀抽取关键帧，返回 PIL Image 列表"""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return frames
        interval = max(1, total // max_frames)
        for i in range(0, min(total, max_frames * interval), interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            if len(frames) >= max_frames:
                break
        cap.release()
    except Exception as e:
        print(f"[Encoder] 视频抽帧失败 {video_path}: {e}")
    return frames


def encode_video(video_path: str, max_frames: int = 8) -> Optional[np.ndarray]:
    """提取视频的平均 Chinese-CLIP 向量（多帧平均）"""
    frames = extract_video_frames(video_path, max_frames)
    if not frames:
        return None
    model, preprocess = get_cn_clip_model()
    vecs = []
    for img in frames:
        try:
            t = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model.encode_image(t)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            vecs.append(feat.cpu().numpy().astype(np.float32)[0])
        except Exception:
            continue
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def describe_video(video_path: str, max_frames: int = 4) -> str:
    """对视频抽帧并生成描述（取第一帧）"""
    frames = extract_video_frames(video_path, max_frames)
    if not frames:
        return ""
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = f.name
        frames[0].save(tmp)
        caption = describe_image(tmp)
        os.unlink(tmp)
        return caption
    except Exception:
        return ""


# ── 文本编码（用于检索，原生支持中文） ─────────────────────────
def encode_text(text: str) -> np.ndarray:
    """将文本（中英文均可）编码为向量，用于语义检索"""
    from cn_clip.clip import tokenize
    model, _ = get_cn_clip_model()
    tokens = tokenize([text]).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)[0]


# ── 批量编码（建索引时加速） ──────────────────────────────────
def encode_images_batch(image_paths: list, batch_size: int = 16) -> list:
    """批量编码图片，返回向量列表（断点续传友好）"""
    model, preprocess = get_cn_clip_model()
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        tensors = []
        valid = []
        for p in batch:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
                valid.append(p)
            except Exception:
                results.append(None)
        if not tensors:
            continue
        batch_tensor = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for p, f in zip(valid, feats):
            results.append(f.cpu().numpy().astype(np.float32)[0])
            # 补齐 skipped
            while len(results) < image_paths.index(p) + 1:
                results.insert(image_paths.index(p), None)
    return results


# ── 文档文本提取 ─────────────────────────────────────────────
import io

def extract_pdf_text(file_path: str, max_pages: int = 20) -> str:
    """提取 PDF 文本内容（支持文本型 PDF）"""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"[Encoder] PDF文本提取失败 {file_path}: {e}")
        return ""

def extract_pdf_images(file_path: str, max_pages: int = 5) -> list:
    """提取 PDF 页面为图像（用于 OCR 扫描版 PDF）"""
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(file_path, dpi=150, first_page=1, last_page=max_pages)
        return images
    except Exception as e:
        print(f"[Encoder] PDF转图像失败 {file_path}: {e}")
        return []

def extract_docx_text(file_path: str) -> str:
    """提取 Word 文档文本内容"""
    try:
        from docx import Document
        doc = Document(file_path)
        text_parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text.strip())
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"[Encoder] Word文本提取失败 {file_path}: {e}")
        return ""

def extract_txt_text(file_path: str) -> str:
    """提取 TXT 文件内容"""
    try:
        # 尝试多种编码
        for enc in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    return f.read(50000).strip()  # 最多读5万字
            except UnicodeDecodeError:
                continue
        return ""
    except Exception as e:
        print(f"[Encoder] TXT文本提取失败 {file_path}: {e}")
        return ""

def extract_xlsx_text(file_path: str, max_rows: int = 200) -> str:
    """提取 Excel 文件文本内容"""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(file_path, read_only=True, data_only=True)
        text_parts = []
        for sheet_name in wb.sheetnames[:3]:  # 最多读3个工作表
            ws = wb[sheet_name]
            row_count = 0
            for row in ws.iter_rows(values_only=True, max_row=max_rows):
                cells = [str(c) for c in row if c is not None and str(c).strip()]
                if cells:
                    text_parts.append(" | ".join(cells))
                row_count += 1
                if row_count >= max_rows:
                    break
        wb.close()
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"[Encoder] Excel文本提取失败 {file_path}: {e}")
        return ""

def extract_document_text(file_path: str) -> str:
    """通用文档文本提取入口，根据扩展名调用对应函数"""
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        return extract_pdf_text(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_docx_text(file_path)
    elif ext == '.txt':
        return extract_txt_text(file_path)
    elif ext in ['.xlsx', '.xls']:
        return extract_xlsx_text(file_path)
    else:
        return ""

def encode_document(file_path: str) -> Optional[np.ndarray]:
    """提取文档文本并用文本编码器编码为向量"""
    try:
        text = extract_document_text(file_path)
        if not text or len(text) < 10:  # 文本太短，跳过
            return None
        # 使用 Chinese-CLIP 的文本编码器
        vec = encode_text(text[:512])  # 只用前512字符编码（避免过长）
        return vec
    except Exception as e:
        print(f"[Encoder] 文档编码失败 {file_path}: {e}")
        return None


if __name__ == "__main__":
    # 快速测试（使用英文避免 Windows 控制台编码问题）
    print("\n── Test text encoding (Chinese) ──")
    vec_cn = encode_text("一只猫坐在沙发上")
    print(f"Chinese vector shape: {vec_cn.shape}")

    print("\n── Test text encoding (English) ──")
    vec_en = encode_text("a cat sitting on a sofa")
    print(f"English vector shape: {vec_en.shape}")

    sim = np.dot(vec_cn, vec_en)
    print(f"Zh/En semantic similarity: {sim:.4f}")
    print("\n[OK] Chinese-CLIP test passed!")
