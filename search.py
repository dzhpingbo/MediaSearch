# -*- coding: utf-8 -*-
"""
search.py - 本地图片/视频/文档语义检索（Chinese-CLIP + BM25 混合版）

功能5: 混合检索 - 向量检索 + BM25 关键词融合，提升检索准确率
功能3: 文档预览 - 搜索结果包含文档内容摘要
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
import math
import pickle
import re
import numpy as np
import faiss
from typing import List, Dict, Optional, Union, Set

# 文件类型常量
FILE_TYPE_ALL = {"image", "video", "document"}
FILE_TYPE_DOCUMENT = {"document"}

# 扩展名到文件类型的映射
EXT_TO_CATEGORY = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".bmp": "image",
    ".gif": "image", ".webp": "image", ".tiff": "image", ".tif": "image",
    ".heic": "image", ".heif": "image",
    ".mp4": "video", ".avi": "video", ".mkv": "video", ".mov": "video",
    ".wmv": "video", ".flv": "video", ".m4v": "video", ".ts": "video",
    ".pdf": "document", ".docx": "document", ".doc": "document",
    ".txt": "document", ".xlsx": "document", ".xls": "document",
}

# 文件类型显示名称
TYPE_DISPLAY_NAMES = {
    "image": "图片",
    "video": "视频",
    "document": "文档",
}


# ══════════════════════════════════════════════════════════════════
#  BM25 核心实现（功能5）
# ══════════════════════════════════════════════════════════════════
class BM25:
    """
    BM25 关键词检索算法
    
    与向量检索互补：
    - 向量检索：语义相似（如"海滩"和"海边"能匹配）
    - BM25：关键词精确匹配（如"合同"必须在文档中出现才被召回）
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = {}      # 词 → 出现文档数
        self.idf = {}            # 词 → IDF 值
        self.doc_len = []        # 每篇文档长度
        self.corpus = []         # 原始文档文本列表

    def _tokenize(self, text: str) -> List[str]:
        """简单中文分词（双字词 + 空格分词）"""
        if not text:
            return []
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', str(text))
        tokens = []
        for word in text.split():
            word = word.strip()
            if not word:
                continue
            # 双字词 + 单字词
            for i in range(len(word)):
                for n in [2, 1]:
                    if i + n <= len(word):
                        tokens.append(word[i:i + n])
        return tokens

    def fit(self, corpus: List[Dict]):
        """
        构建 BM25 索引
        
        Args:
            corpus: [{"doc_id": str, "text": str, "path": str, ...}, ...]
        """
        self.corpus = corpus
        self.corpus_size = len(corpus)
        nd = {}  # 词 → 出现文档数

        for doc in corpus:
            doc_text = doc.get("text", "")
            self.doc_len.append(len(doc_text))
            tokens = self._tokenize(doc_text)
            freq = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            for token, f in freq.items():
                if token not in nd:
                    nd[token] = 0
                nd[token] += 1

        # 计算 IDF
        for token, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[token] = idf

        # 平均文档长度
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 1

    def score(self, query: str, doc_idx: int) -> float:
        """计算 query 对第 doc_idx 篇文档的 BM25 得分"""
        if doc_idx < 0 or doc_idx >= self.corpus_size:
            return 0.0
        
        doc_text = self.corpus[doc_idx].get("text", "")
        doc_len = self.doc_len[doc_idx]
        tokens = self._tokenize(doc_text)
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        score = 0.0
        q_tokens = self._tokenize(query)
        for token in q_tokens:
            if token not in freq:
                continue
            tf = freq[token]
            idf = self.idf.get(token, 0)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
        return score

    def search(self, query: str, doc_ids: List[str] = None) -> List[tuple]:
        """
        BM25 搜索
        
        Args:
            query: 查询词
            doc_ids: 可选，限制在哪些 doc_id 中搜索
        
        Returns:
            [(doc_id, score), ...] 按得分降序
        """
        if not self.corpus_size:
            return []
        
        # 构建 doc_id → index 映射
        doc_id_to_idx = {}
        for i, doc in enumerate(self.corpus):
            if doc_ids is None or doc.get("doc_id") in doc_ids:
                doc_id_to_idx[doc.get("doc_id", i)] = i

        scores = []
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        for doc_id, idx in doc_id_to_idx.items():
            s = self.score(query, idx)
            if s > 0:
                scores.append((doc_id, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ══════════════════════════════════════════════════════════════════
#  检索引擎
# ══════════════════════════════════════════════════════════════════

class MediaSearchEngine:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index = None
        self.metadata = []
        self.bm25 = None
        self.bm25_corpus = []   # [{"doc_id": ..., "text": ..., ...}, ...]
        self._load()

    def _load(self):
        """加载向量索引 + BM25 索引"""
        index_path = os.path.join(self.index_dir, "vectors.faiss")
        meta_path = os.path.join(self.index_dir, "metadata.json")
        bm25_path = os.path.join(self.index_dir, "bm25_index.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引不存在: {index_path}\n请先运行 build_index.py 建立索引")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # 加载 BM25 索引（功能5: 混合检索）
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "rb") as f:
                    bm25_data = pickle.load(f)
                    self.bm25_corpus = bm25_data.get("corpus", [])
                if self.bm25_corpus:
                    self.bm25 = BM25()
                    self.bm25.fit(self.bm25_corpus)
                    print(f"[Search] 索引加载完成：{len(self.metadata)} 条（向量）+ {len(self.bm25_corpus)} 个文档（BM25）")
                else:
                    print(f"[Search] 索引加载完成：{len(self.metadata)} 条（向量）")
            except Exception as e:
                print(f"[Search] BM25 索引加载失败: {e}，将仅使用向量检索")
                self.bm25 = None
                print(f"[Search] 索引加载完成：{len(self.metadata)} 条（向量）")
        else:
            print(f"[Search] 索引加载完成：{len(self.metadata)} 条（向量），无 BM25 索引")

    def search(self, query: str, top_k: int = 20,
               file_types: Optional[Union[str, Set[str]]] = None,
               min_score: float = 0.15,
               hybrid_weight: float = 0.4) -> List[Dict]:
        """
        功能5: 混合检索 - 向量检索 + BM25 关键词融合
        
        Args:
            query: 自然语言查询（中英文均可）
            top_k: 返回结果数量
            file_types: 过滤类型（None = 所有类型）
            min_score: 最低相似度阈值（0~1）
            hybrid_weight: BM25 权重（0.0~1.0），默认 0.4
                         - 0.0: 仅向量检索
                         - 1.0: 仅 BM25 检索
                         - 0.4: 向量 60% + BM25 40%（推荐）
        """
        from core_encoder import encode_text

        # 解析 file_types
        if file_types is None or file_types == "all":
            file_types_set = None
        elif isinstance(file_types, str):
            file_types_set = {file_types}
        else:
            file_types_set = file_types

        # ── Step 1: 向量检索 ──────────────────────────────
        vec = encode_text(query).reshape(1, -1).astype(np.float32)
        k = min(top_k * 3, self.index.ntotal)
        if k == 0:
            return []
        scores, ids = self.index.search(vec, k)

        # 组装向量检索结果
        vec_results = {}   # doc_id → best chunk result
        for score, idx in zip(scores[0], ids[0]):
            if score < min_score * 0.5:  # 放宽阈值，后面再过滤
                continue
            if 0 <= idx < len(self.metadata):
                m = self.metadata[idx].copy()
                m["vec_score"] = float(score)
                m["bm25_score"] = 0.0
                m["score"] = float(score)  # 临时
                doc_id = m.get("doc_id", m.get("path"))

                # 保留每个 doc_id 得分最高的 chunk
                if doc_id not in vec_results or score > vec_results[doc_id]["vec_score"]:
                    vec_results[doc_id] = m

        # ── Step 2: BM25 检索（仅对文档）────────────────
        bm25_scores = {}
        if self.bm25 is not None and hybrid_weight > 0:
            bm25_raw = self.bm25.search(query)
            # 归一化 BM25 得分（最大值为 1）
            if bm25_raw:
                max_bm25 = max(s for _, s in bm25_raw)
                for doc_id, s in bm25_raw:
                    if max_bm25 > 0:
                        bm25_scores[doc_id] = s / max_bm25

        # ── Step 3: 混合评分 ─────────────────────────────
        final_results = []
        all_doc_ids = set(vec_results.keys()) | set(bm25_scores.keys())

        for doc_id in all_doc_ids:
            vec_score = vec_results.get(doc_id, {})
            bm25_s = bm25_scores.get(doc_id, 0.0)

            # 如果该 doc_id 在向量结果中，取最佳 chunk
            if vec_score:
                m = vec_score
                final_vec_score = m["vec_score"]
                best_chunk_text = m.get("chunk_text", m.get("doc_text", ""))[:500]
                caption = m.get("caption", "")
            else:
                # 仅 BM25 召回（不在向量索引中的文档？）
                m = {}
                final_vec_score = 0.0
                best_chunk_text = ""
                caption = ""

            # 混合得分 = (1-w) * vec_score + w * bm25_score
            hybrid_score = (1 - hybrid_weight) * final_vec_score + hybrid_weight * bm25_s

            # 图片/视频类文件不参与 BM25 混合
            ftype = m.get("type", "unknown")
            if ftype in ("image", "video"):
                hybrid_score = final_vec_score

            if hybrid_score < min_score:
                continue

            result = {
                **m,
                "score": round(hybrid_score, 4),
                "vec_score": round(final_vec_score, 4),
                "bm25_score": round(bm25_s, 4),
                "chunk_text": best_chunk_text,  # 功能3: 文档预览文本
            }
            final_results.append(result)

        # 类型过滤
        if file_types_set:
            final_results = [r for r in final_results if r.get("type") in file_types_set]

        # 排序
        final_results.sort(key=lambda x: x["score"], reverse=True)

        # 去重（每个 doc_id 只保留一个）
        seen = set()
        deduped = []
        for r in final_results:
            doc_id = r.get("doc_id", r.get("path"))
            if doc_id not in seen:
                seen.add(doc_id)
                deduped.append(r)

        return deduped[:top_k]

    def search_similar(self, image_path: str, top_k: int = 10,
                       file_types: Optional[Union[str, Set[str]]] = None) -> List[Dict]:
        """以图搜图：找和给定图片最相似的文件"""
        from core_encoder import encode_image

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
                if m.get("path") != image_path:
                    results.append(m)

        if file_types_set:
            results = [r for r in results if r.get("type") in file_types_set]

        return results[:top_k]

    def search_by_ext(self, query: str, extensions: List[str], top_k: int = 20) -> List[Dict]:
        """按扩展名搜索（更精细的控制）"""
        results = self.search(query, top_k=top_k * 2, min_score=0.0)
        filtered = [r for r in results if r.get("ext", "").lower() in extensions]
        return filtered[:top_k]

    def get_document_preview(self, doc_path: str, max_chars: int = 300) -> str:
        """功能3: 获取文档预览文本（用于 UI 展示）"""
        for m in self.metadata:
            if m.get("path") == doc_path or m.get("doc_id") == doc_path:
                text = m.get("chunk_text", m.get("doc_text", ""))
                if text:
                    return text[:max_chars]
        return ""

    def stats(self) -> Dict:
        """返回索引统计信息"""
        images = sum(1 for m in self.metadata if m.get("type") == "image")
        videos = sum(1 for m in self.metadata if m.get("type") == "video")
        docs = sum(1 for m in self.metadata if m.get("type") == "document")
        # 去重：统计独立文档数
        doc_ids = set(m.get("doc_id") for m in self.metadata if m.get("type") == "document")
        return {
            "total": len(self.metadata),
            "images": images,
            "videos": videos,
            "documents": len(doc_ids),
            "bm25_docs": len(self.bm25_corpus),
            "index_dir": self.index_dir
        }


# ── 命令行快速检索 ───────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="语义检索本地图片/视频/文档（混合检索版）")
    parser.add_argument("query", help="搜索关键词（中英文均可）")
    parser.add_argument("--index", default=r"E:\dzhwork\pictureVedioIndex", help="索引目录")
    parser.add_argument("--top", type=int, default=10, help="返回结果数量")
    parser.add_argument("--types", default="all",
                        help="文件类型过滤，逗号分隔（image,video,document）或 all")
    parser.add_argument("--hybrid", type=float, default=0.4,
                        help="BM25 混合权重（0.0~1.0），默认 0.4（推荐）")
    parser.add_argument("--open", action="store_true", help="自动打开第一个结果")
    args = parser.parse_args()

    if args.types == "all":
        file_types = None
    else:
        file_types = set(args.types.split(","))

    engine = MediaSearchEngine(args.index)
    results = engine.search(args.query, top_k=args.top, file_types=file_types,
                           hybrid_weight=args.hybrid)

    if not results:
        print("未找到相关文件，请尝试其他关键词")
    else:
        print(f"\n找到 {len(results)} 个相关文件（混合检索，BM25权重={args.hybrid}）：\n")
        for i, r in enumerate(results, 1):
            type_icon = {"image": "IMG", "video": "VID", "document": "DOC"}.get(r.get("type"), "???")
            type_name = TYPE_DISPLAY_NAMES.get(r.get("type"), r.get("type"))
            vec_s = r.get("vec_score", 0)
            bm25_s = r.get("bm25_score", 0)
            print(f"{i:2d}. [{r['score']:.3f}] [{type_icon}] {type_name} | {r['filename']}")
            print(f"    VEC:{vec_s:.3f} BM25:{bm25_s:.3f} | {r.get('size_mb', '?')}MB | {r.get('mtime', '?')}")
            print(f"    {r['path']}")
            # 功能3: 文档预览
            chunk_text = r.get("chunk_text", "")
            if chunk_text:
                preview = chunk_text[:100].replace("\n", " ")
                print(f"    Preview: {preview}...")
            print()
