#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_index.py - 扫描本地图片/视频/文档，构建向量知识库（Chinese-CLIP 版）
用法：
  python build_index.py                          # 扫描常用目录，索引存到默认位置
  python build_index.py --dirs "D:\Photos" "E:\Docs"  # 自定义目录
  python build_index.py --resume                # 只处理未完成的文件
  python build_index.py --types image,video,document  # 指定文件类型
"""
import os
# 代理设置（必须在所有 import 之前）
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import sys
import json
import time
import argparse
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime

# ── 配置 ────────────────────────────────────────────────────────────────────
# 默认索引保存目录（用户指定：E盘，防止C盘空间不足）
DEFAULT_OUTPUT = r"E:\dzhwork\pictureVedioIndex"

# 默认扫描目录（Windows 常见图片/视频/文档位置）
DEFAULT_DIRS = [
    r"C:\Users\Administrator\Pictures",
    r"C:\Users\Administrator\Videos",
    r"C:\Users\Administrator\Desktop",
    r"C:\Users\Administrator\Documents",
]

# 支持的文件格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif", ".heic", ".heif", ".jfif"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".m4v", ".ts", ".rmvb", ".m2ts", ".mpg", ".mpeg"}
DOCUMENT_EXTS = {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls"}
ALL_EXTS = IMAGE_EXTS | VIDEO_EXTS | DOCUMENT_EXTS

# 文件类型 mapping
EXT_TO_TYPE = {}
for ext in IMAGE_EXTS:
    EXT_TO_TYPE[ext] = "image"
for ext in VIDEO_EXTS:
    EXT_TO_TYPE[ext] = "video"
for ext in DOCUMENT_EXTS:
    EXT_TO_TYPE[ext] = "document"


def scan_files(dirs: list, max_files: int = None, file_types: set = None) -> list:
    """
    扫描目录下所有图片/视频/文档文件（去重、排序）
    file_types: 指定要扫描的文件类型集合，如 {"image", "video", "document"}
    """
    files = []
    seen = set()
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"[扫描] 目录不存在，跳过: {d}")
            continue
        print(f"[扫描] {d}")
        count = 0
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext not in ALL_EXTS:
                continue
            # 按文件类型过滤
            ftype = EXT_TO_TYPE.get(ext, "unknown")
            if file_types and ftype not in file_types:
                continue
            fstr = str(f)
            if fstr not in seen:
                seen.add(fstr)
                files.append({"path": fstr, "type": ftype, "ext": ext})
                count += 1
        print(f"       发现 {count} 个文件")
    print(f"\n[扫描] 共发现 {len(files)} 个文件（已去重）")
    return files


def load_existing_index(output_dir: str):
    """加载已有索引（断点续传）"""
    meta_path = os.path.join(output_dir, "metadata.json")
    index_path = os.path.join(output_dir, "vectors.faiss")
    if os.path.exists(meta_path) and os.path.exists(index_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        index = faiss.read_index(index_path)
        print(f"[索引] 加载已有索引：{len(metadata)} 条记录")
        return index, metadata
    return None, []


def save_index(index, metadata, output_dir: str):
    """保存向量索引和元数据"""
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "vectors.faiss"))
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[索引] 已保存：{len(metadata)} 条")


def build_index(dirs: list, output_dir: str, use_caption: bool = False,
                batch_save: int = 50, max_files: int = None,
                file_types: set = None):
    """
    主函数：扫描并建立向量索引（Chinese-CLIP 向量）
    file_types: 指定要处理的文件类型，如 {"image", "video", "document"}
    """
    from core_encoder import (
        encode_image, encode_video, encode_document,
        describe_image, describe_video, extract_document_text
    )

    os.makedirs(output_dir, exist_ok=True)
    all_files = scan_files(dirs, max_files, file_types)

    if max_files and len(all_files) > max_files:
        print(f"[索引] 限制处理前 {max_files} 个文件（测试模式）")
        all_files = all_files[:max_files]

    # 加载已有索引（断点续传）
    index, metadata = load_existing_index(output_dir)
    done_paths = {m["path"] for m in metadata}

    # 过滤已处理文件
    todo = [f for f in all_files if f["path"] not in done_paths]
    print(f"[索引] 待处理：{len(todo)} 个（已完成 {len(done_paths)} 个）")

    if not todo:
        print("[索引] 所有文件已处理完成！")
        return index, metadata

    # 初始化 FAISS 索引（Chinese-CLIP ViT-L-14 向量维度 = 768）
    DIM = 768
    if index is None:
        index = faiss.IndexFlatIP(DIM)   # 内积 = 余弦相似度（向量已归一化）

    new_vecs = []
    new_meta = []
    processed = 0
    failed = 0

    start = time.time()
    for i, f in enumerate(todo):
        path = f["path"]
        ftype = f["type"]

        try:
            vec = None
            caption = ""
            doc_text = ""

            if ftype == "image":
                vec = encode_image(path)
                caption = describe_image(path) if use_caption else ""
            elif ftype == "video":
                vec = encode_video(path)
                caption = describe_video(path) if use_caption else ""
            elif ftype == "document":
                # 文档：提取文本并编码
                doc_text = extract_document_text(path)
                vec = encode_document(path)
                caption = doc_text[:200] if doc_text else ""  # 前200字符作为描述

            if vec is None:
                failed += 1
                continue

            new_vecs.append(vec.astype(np.float32))

            stat = os.stat(path)
            new_meta.append({
                "path": path,
                "type": ftype,
                "ext": f["ext"],
                "caption": caption,
                "doc_text": doc_text[:1000] if doc_text else "",  # 存储前1000字符用于搜索
                "filename": os.path.basename(path),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "index_id": len(metadata) + len(new_meta)
            })
            processed += 1

        except Exception as e:
            print(f"[索引] 处理失败 {path}: {e}")
            failed += 1

        # 进度报告（每10个文件）
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - i - 1) / rate if rate > 0 else 0
            print(f"[进度] {i+1}/{len(todo)} | "
                  f"成功:{processed} 失败:{failed} | "
                  f"速度:{rate:.1f}张/s | 剩余约:{remaining/60:.1f}分钟")

        # 定期保存（断点续传）
        if len(new_vecs) >= batch_save:
            mat = np.stack(new_vecs)
            index.add(mat)
            metadata.extend(new_meta)
            save_index(index, metadata, output_dir)
            new_vecs.clear()
            new_meta.clear()

    # 保存剩余
    if new_vecs:
        mat = np.stack(new_vecs)
        index.add(mat)
        metadata.extend(new_meta)
        save_index(index, metadata, output_dir)

    total_time = time.time() - start
    print(f"\n[完成] 总计 {len(metadata)} 条记录，耗时 {total_time/60:.1f} 分钟")
    print(f"[完成] 本次成功: {processed}，失败: {failed}")
    print(f"[完成] 索引保存至: {output_dir}")
    return index, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建本地图片/视频/文档向量索引（Chinese-CLIP）")
    parser.add_argument("--dirs", nargs="+", help="要扫描的目录列表（不填则扫描常用目录）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"索引保存目录（默认: {DEFAULT_OUTPUT}）")
    parser.add_argument("--caption", action="store_true", help="同时生成图片描述（更慢但检索更准）")
    parser.add_argument("--batch", type=int, default=50, help="每N个文件保存一次（断点续传，默认50）")
    parser.add_argument("--max-files", type=int, default=None, help="仅处理前N个文件（测试用）")
    parser.add_argument("--types", default="image,video,document",
                        help="要索引的文件类型，逗号分隔（image,video,document），默认全部")
    args = parser.parse_args()

    dirs = args.dirs if args.dirs else DEFAULT_DIRS

    # 解析文件类型
    file_types = set(args.types.split(","))
    # 验证文件类型
    valid_types = {"image", "video", "document"}
    file_types = file_types & valid_types
    if not file_types:
        file_types = valid_types

    print("=" * 60)
    print("MediaSearch 索引构建")
    print("=" * 60)
    print(f"扫描目录: {dirs}")
    print(f"索引输出: {args.output}")
    print(f"文件类型: {file_types}")
    print(f"生成描述: {args.caption}")
    print("=" * 60)

    build_index(dirs, args.output, use_caption=args.caption,
                batch_save=args.batch, max_files=args.max_files,
                file_types=file_types)
