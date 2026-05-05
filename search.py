# -*- coding: utf-8 -*-
"""
search.py - 自然语言检索图片/视频/文档（Chinese-CLIP 版）
支持中英文混合查询，返回最相关文件路径和分数
支持按文件类型过滤（图片/视频/PDF/Word/TXT/XLSX）
"""
# 禁用代理（必须在 import 之前）
import os as _os
_os.environ["http_proxy"] = ""
_os.environ["https_proxy"] = ""
_os.environ["HTTP_PROXY"] = ""
_os.environ["HTTPS_PROXY"] = ""
_os.environ["no_proxy"] = "*"
_os.environ["NO_PROXY"] = "*"

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Optional, Union, Set

# 文件类型常量
FILE_TYPE_ALL = {"image", "video", "document"}
FILE_TYPE_IMAGE = {"image"}
FILE_TYPE_VIDEO = {"video"}
FILE_TYPE_DOCUMENT = {"document"}

# 扩展名到文件类型的映射（详细版，用于 UI 筛选）
EXT_TO_CATEGORY = {
    # 图片
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".bmp": "image",
    ".gif": "image", ".webp": "image", ".tiff": "image", ".tif": "image",
    ".heic": "image", ".heif": "image",
    # 视频
    ".mp4": "video", ".avi": "video", ".mkv": "video", ".mov": "video",
    ".wmv": "video", ".flv": "video", ".m4v": "video", ".ts": "video",
    # 文档
    ".pdf": "document", ".docx": "document", ".doc": "document",
    ".txt": "document", ".xlsx": "document", ".xls": "document",
}

# 文件类型显示名称
TYPE_DISPLAY_NAMES = {
    "image": "图片",
    "video": "视频",
    "document": "文档",
}


class MediaSearchEngine:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index = None
        self.metadata = []
        self._load()

    def _load(self):
        """加载向量索引"""
        index_path = os.path.join(self.index_dir, "vectors.faiss")
        meta_path = os.path.join(self.index_dir, "metadata.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引不存在: {index_path}\n请先运行 build_index.py 建立索引")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[Search] 索引加载完成：{len(self.metadata)} 个文件")

    def search(self, query: str, top_k: int = 20,
               file_types: Optional[Union[str, Set[str]]] = None,
               min_score: float = 0.15) -> List[Dict]:
        """
        语义检索

        Args:
            query: 自然语言查询（中英文均可）
            top_k: 返回结果数量
            file_types: 过滤类型，可以是：
                - None 或 "all"：返回所有类型
                - "image" / "video" / "document"：单个类型
                - {"image", "document"}：多个类型
            min_score: 最低相似度阈值（0~1）

        Returns:
            结果列表，每项含 path/type/score/caption/filename
        """
        from core_encoder import encode_text

        # 解析 file_types 参数
        if file_types is None or file_types == "all":
            file_types_set = None
        elif isinstance(file_types, str):
            file_types_set = {file_types}
        else:
            file_types_set = file_types

        # 直接用原始查询编码（Chinese-CLIP 原生支持中文）
        vec = encode_text(query).reshape(1, -1).astype(np.float32)
        k = min(top_k * 2, self.index.ntotal)
        if k == 0:
            return []
        scores, ids = self.index.search(vec, k)

        # 组装结果
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if score < min_score:
                continue
            if 0 <= idx < len(self.metadata):
                m = self.metadata[idx].copy()
                m["score"] = round(float(score), 4)
                results.append(m)

        # 类型过滤
        if file_types_set:
            results = [r for r in results if r.get("type") in file_types_set]

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)

        # 同时，对文档类型结果，做文本匹配加分（提高准确率）
        if file_types_set is None or "document" in file_types_set:
            results = self._boost_by_text_match(results, query)

        return results[:top_k]

    def _boost_by_text_match(self, results: List[Dict], query: str) -> List[Dict]:
        """对文档结果，如果文本内容包含查询词，提高排序"""
        query_lower = query.lower()
        for r in results:
            if r.get("type") == "document" and r.get("doc_text"):
                doc_text_lower = r["doc_text"].lower()
                # 如果文档文本包含查询词，加分
                if query_lower in doc_text_lower:
                    r["score"] = min(1.0, r["score"] + 0.1)
                # 如果是短语匹配，再加分
                if len(query_lower) > 2 and query_lower in doc_text_lower:
                    r["score"] = min(1.0, r["score"] + 0.05)
        # 重新排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def search_similar(self, image_path: str, top_k: int = 10,
                       file_types: Optional[Union[str, Set[str]]] = None) -> List[Dict]:
        """以图搜图：找和给定图片最相似的文件"""
        from core_encoder import encode_image

        # 解析 file_types 参数
        if file_types is None or file_types == "all":
            file_types_set = None
        elif isinstance(file_types, str):
            file_types_set = {file_types}
        else:
            file_types_set = file_types

        vec = encode_image(image_path)
        if vec is None:
            return []
        vec = vec.reshape(1, -1).astype(np.float32)
        k = min(top_k + 1, self.index.ntotal)
        scores, ids = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(self.metadata):
                m = self.metadata[idx].copy()
                m["score"] = round(float(score), 4)
                if m["path"] != image_path:
                    results.append(m)

        # 类型过滤
        if file_types_set:
            results = [r for r in results if r.get("type") in file_types_set]

        return results[:top_k]

    def search_by_ext(self, query: str, extensions: List[str], top_k: int = 20) -> List[Dict]:
        """
        按扩展名搜索（更精细的控制）
        例如：extensions=[".pdf", ".docx"] 只搜 PDF 和 Word 文档
        """
        results = self.search(query, top_k=top_k * 2, min_score=0.0)
        filtered = [r for r in results if r.get("ext", "").lower() in extensions]
        return filtered[:top_k]

    def stats(self) -> Dict:
        """返回索引统计信息"""
        images = sum(1 for m in self.metadata if m.get("type") == "image")
        videos = sum(1 for m in self.metadata if m.get("type") == "video")
        documents = sum(1 for m in self.metadata if m.get("type") == "document")
        return {
            "total": len(self.metadata),
            "images": images,
            "videos": videos,
            "documents": documents,
            "index_dir": self.index_dir
        }


# ── 命令行快速检索 ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="语义检索本地图片/视频/文档")
    parser.add_argument("query", help="搜索关键词（中英文均可）")
    parser.add_argument("--index", default=r"E:\dzhwork\pictureVedioIndex", help="索引目录")
    parser.add_argument("--top", type=int, default=10, help="返回结果数量")
    parser.add_argument("--types", default="all",
                        help="文件类型过滤，逗号分隔（image,video,document）或 all")
    parser.add_argument("--open", action="store_true", help="自动打开第一个结果")
    args = parser.parse_args()

    # 解析文件类型
    if args.types == "all":
        file_types = None
    else:
        file_types = set(args.types.split(","))

    engine = MediaSearchEngine(args.index)
    results = engine.search(args.query, top_k=args.top, file_types=file_types)

    if not results:
        print("未找到相关文件，请尝试其他关键词")
    else:
        print(f"\n找到 {len(results)} 个相关文件：\n")
        for i, r in enumerate(results, 1):
            type_icon = {"image": "🖼️", "video": "🎬", "document": "📄"}.get(r.get("type"), "❓")
            type_name = TYPE_DISPLAY_NAMES.get(r.get("type"), r.get("type"))
            caption = f" | {r['caption']}" if r.get("caption") else ""
            print(f"{i:2d}. [{r['score']:.3f}] {type_icon} [{type_name}] {r['path']}{caption}")
            print(f"      {r['filename']} | {r.get('size_mb', '?')}MB | {r.get('mtime', '?')}")
            # 如果是文档，显示部分文本内容
            if r.get("type") == "document" and r.get("doc_text"):
                preview = r["doc_text"][:100].replace("\n", " ")
                print(f"      内容预览: {preview}...")

        if args.open and results:
            first = results[0]["path"]
            print(f"\n打开: {first}")
            os.startfile(first)
