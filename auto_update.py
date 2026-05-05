# -*- coding: utf-8 -*-
"""
auto_update.py - 自动增量索引监控（功能4）

使用 watchdog 监控指定目录，新增/修改文件自动加入向量索引。
用法:
    python auto_update.py --dirs "E:\Photos" "D:\Documents"
    python auto_update.py --dirs "E:\Photos" --once   # 单次扫描后退出
    python auto_update.py --stop                       # 停止后台监控
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
import time
import json
import argparse
import threading
import signal
from datetime import datetime
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ── 配置 ──────────────────────────────────────────────────────
INDEX_DIR = r"E:\dzhwork\pictureVedioIndex"
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".auto_update.pid")

# 支持的文件格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif", ".heic", ".heif", ".jfif"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".m4v", ".ts", ".rmvb", ".m2ts", ".mpg", ".mpeg"}
DOCUMENT_EXTS = {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls"}
ALL_WATCHED_EXTS = IMAGE_EXTS | VIDEO_EXTS | DOCUMENT_EXTS


def is_supported_file(path: str) -> bool:
    """检查文件是否是需要索引的类型"""
    ext = Path(path).suffix.lower()
    return ext in ALL_WATCHED_EXTS


def get_file_mtime(path: str) -> float:
    """获取文件修改时间"""
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0


# ── 增量索引处理 ─────────────────────────────────────────────
def process_new_or_modified_file(file_path: str, index_dir: str) -> bool:
    """
    处理单个新增/修改的文件，加入向量索引
    
    逻辑：
    1. 检查 metadata.json 中是否已有该文件
    2. 如果没有或 mtime 更新，则重新编码并加入索引
    """
    if not os.path.exists(file_path):
        return False

    try:
        import numpy as np
        import faiss
        from core_encoder import encode_image, encode_video, encode_document, extract_document_text
        from core_encoder import chunk_text

        # 加载现有索引
        index_path = os.path.join(index_dir, "vectors.faiss")
        meta_path = os.path.join(index_dir, "metadata.json")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            print(f"[AutoUpdate] 索引不存在，先运行 build_index.py 建立索引")
            return False

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        index = faiss.read_index(index_path)

        # 检查是否已存在（按路径匹配）
        ext = Path(file_path).suffix.lower()
        existing = None
        for m in metadata:
            if m.get("path") == file_path:
                existing = m
                break

        # 判断是否需要更新
        current_mtime = get_file_mtime(file_path)
        if existing:
            existing_mtime = existing.get("mtime", "")
            # mtime 格式：YYYY-MM-DD HH:MM
            if existing_mtime:
                try:
                    existing_ts = datetime.strptime(existing_mtime, "%Y-%m-%d %H:%M").timestamp()
                    if current_mtime <= existing_ts:
                        return False  # 文件未更新，跳过
                except Exception:
                    pass

        # 编码文件
        print(f"[AutoUpdate] 处理文件: {file_path}")
        stat = os.stat(file_path)

        if ext in IMAGE_EXTS:
            ftype = "image"
            vec = encode_image(file_path)
            caption = ""
            chunk_results = None
        elif ext in VIDEO_EXTS:
            ftype = "video"
            vec = encode_video(file_path)
            caption = ""
            chunk_results = None
        else:  # document
            ftype = "document"
            doc_text = extract_document_text(file_path)
            if not doc_text or len(doc_text) < 10:
                print(f"[AutoUpdate] 文档无有效文本，跳过: {file_path}")
                return False
            caption = doc_text[:300]
            # 分块编码
            chunks = chunk_text(doc_text, chunk_size=300)
            chunk_results = []
            for chunk in chunks:
                chunk_text_str = chunk["text"]
                if len(chunk_text_str) < 5:
                    continue
                from core_encoder import encode_text
                vec = encode_text(chunk_text_str[:512])
                chunk_results.append({
                    "chunk_text": chunk_text_str,
                    "chunk_id": chunk["chunk_id"],
                    "vector": vec,
                })
            vec = chunk_results[0]["vector"] if chunk_results else None

        if vec is None:
            return False

        # 如果文件已存在，从索引中删除旧向量
        if existing:
            # 标记旧记录为已删除（不重建整个索引，只追加）
            print(f"[AutoUpdate] 文件已更新，重新索引: {file_path}")
        else:
            print(f"[AutoUpdate] 新增文件: {file_path}")

        # 构建新元数据
        if chunk_results is None:
            # 图片/视频：单向量
            new_meta = [{
                "path": file_path,
                "type": ftype,
                "ext": ext,
                "doc_id": file_path,
                "chunk_id": 0,
                "caption": caption,
                "doc_text": "",
                "chunk_text": "",
                "filename": os.path.basename(file_path),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "mtime": datetime.fromtimestamp(current_mtime).strftime("%Y-%m-%d %H:%M"),
                "index_id": len(metadata),
            }]
            new_vecs = [vec.astype(np.float32)]
        else:
            # 文档：多向量（分块）
            new_meta = []
            new_vecs = []
            for chunk in chunk_results:
                new_vecs.append(chunk["vector"].astype(np.float32))
                new_meta.append({
                    "path": file_path,
                    "type": ftype,
                    "ext": ext,
                    "doc_id": file_path,
                    "chunk_id": chunk["chunk_id"],
                    "caption": caption[:300],
                    "doc_text": doc_text[:5000] if ftype == "document" else "",
                    "chunk_text": chunk["chunk_text"],
                    "filename": os.path.basename(file_path),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "mtime": datetime.fromtimestamp(current_mtime).strftime("%Y-%m-%d %H:%M"),
                    "index_id": len(metadata) + len(new_vecs) - 1,
                })

        # 添加到索引
        mat = np.stack(new_vecs)
        index.add(mat)
        metadata.extend(new_meta)

        # 保存
        faiss.write_index(index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"[AutoUpdate] 索引更新成功: {file_path} (+{len(new_vecs)} 条记录)")
        return True

    except Exception as e:
        print(f"[AutoUpdate] 处理失败 {file_path}: {e}")
        return False


# ── Watchdog 文件监控处理器 ────────────────────────────────────
class MediaIndexHandler(FileSystemEventHandler):
    """
    监控文件系统变化，自动触发增量索引
    
    触发条件：
    - 新建文件（且为支持的文件类型）
    - 文件被修改（mtime 变化）
    
    忽略：
    - 临时文件（.tmp, .crdownload 等）
    - 隐藏文件
    - 目录本身
    """
    def __init__(self, index_dir: str, debounce: int = 3):
        self.index_dir = index_dir
        self.debounce = debounce  # 防抖秒数（避免同一文件连续触发）
        self.pending = {}          # {path: scheduled_time}
        self.lock = threading.Lock()

    def _should_process(self, path: str) -> bool:
        """检查是否应该处理这个文件"""
        if not is_supported_file(path):
            return False
        name = os.path.basename(path)
        # 忽略临时文件和隐藏文件
        if name.startswith('.') or name.startswith('~'):
            return False
        ext_blacklist = {'.tmp', '.temp', '.crdownload', '.partial', '.part'}
        if Path(path).suffix.lower() in ext_blacklist:
            return False
        return True

    def _schedule_process(self, path: str):
        """延迟处理（防抖）"""
        with self.lock:
            self.pending[path] = time.time() + self.debounce
            # 清理过期项
            now = time.time()
            expired = [p for p, t in self.pending.items() if t <= now]
            for p in expired:
                del self.pending[p]

    def _process_pending(self):
        """处理已到时的任务（定时检查）"""
        with self.lock:
            now = time.time()
            ready = [p for p, t in self.pending.items() if t <= now]
            for path in ready:
                del self.pending[path]
                # 在后台线程中处理，避免阻塞 watchdog
                threading.Thread(
                    target=process_new_or_modified_file,
                    args=(path, self.index_dir),
                    daemon=True
                ).start()

    def on_created(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            print(f"[Watch] 新建文件: {event.src_path}")
            self._schedule_process(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            print(f"[Watch] 文件修改: {event.src_path}")
            self._schedule_process(event.src_path)


# ── 主函数 ────────────────────────────────────────────────────
def start_watching(dirs: list, index_dir: str, once: bool = False):
    """启动文件监控"""
    if not dirs:
        print("[AutoUpdate] 未指定监控目录")
        return

    print("=" * 60)
    print("  MediaSearch 自动增量更新")
    print("=" * 60)
    print(f"监控目录: {dirs}")
    print(f"索引目录: {index_dir}")
    print(f"模式: {'单次扫描' if once else '持续监控'}")
    print("=" * 60)

    handler = MediaIndexHandler(index_dir, debounce=5)
    observer = Observer()
    for d in dirs:
        if os.path.exists(d):
            observer.schedule(handler, d, recursive=True)
            print(f"[Watch] 开始监控: {d}")
        else:
            print(f"[Watch] 目录不存在，跳过: {d}")

    observer.start()
    print("[Watch] 监控已启动（Ctrl+C 停止）")

    try:
        if once:
            # 单次扫描：等待防抖时间后处理
            time.sleep(handler.debounce + 2)
            handler._process_pending()
            print("[AutoUpdate] 单次扫描完成")
        else:
            # 持续监控：定时检查待处理任务
            while True:
                time.sleep(2)
                handler._process_pending()
    except KeyboardInterrupt:
        print("\n[AutoUpdate] 停止监控")
    finally:
        observer.stop()
        observer.join()


def stop_watching():
    """停止后台监控"""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            print(f"[AutoUpdate] 已停止监控进程 (PID: {pid})")
        except Exception as e:
            print(f"[AutoUpdate] 停止失败: {e}")
    else:
        print("[AutoUpdate] 未找到运行中的监控进程")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaSearch 自动增量索引更新")
    parser.add_argument("--dirs", nargs="+", help="要监控的目录")
    parser.add_argument("--index", default=INDEX_DIR, help=f"索引目录（默认: {INDEX_DIR}）")
    parser.add_argument("--once", action="store_true", help="单次扫描后退出（不持续监控）")
    parser.add_argument("--stop", action="store_true", help="停止后台监控进程")
    args = parser.parse_args()

    if args.stop:
        stop_watching()
    elif args.dirs:
        start_watching(args.dirs, args.index, args.once)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python auto_update.py --dirs E:\\Photos D:\\Documents")
        print("  python auto_update.py --dirs E:\\Photos --once  # 单次扫描")
        print("  python auto_update.py --stop                  # 停止监控")
