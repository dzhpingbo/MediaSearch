"""
app.py - 本地图片/视频/文档语义检索 Web UI（Chinese-CLIP + BM25 混合检索版）
运行: python app.py
然后浏览器访问 http://localhost:7860

新增功能：
- 功能3: 文档预览（搜索结果直接展示内容摘要）
- 功能5: BM25 混合检索权重控制
"""
# 禁用代理（必须在 import os 之前）
import os as _os
_os.environ["http_proxy"] = ""
_os.environ["https_proxy"] = ""
_os.environ["HTTP_PROXY"] = ""
_os.environ["HTTPS_PROXY"] = ""
_os.environ["no_proxy"] = "*"
_os.environ["NO_PROXY"] = "*"

import os
import json
import gradio as gr
from pathlib import Path

INDEX_DIR = r"E:\dzhwork\pictureVedioIndex"

# ── 全局引擎（延迟加载） ──────────────────────────────────────
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        from search import MediaSearchEngine
        _engine = MediaSearchEngine(INDEX_DIR)
    return _engine


# ── 文件类型配置 ──────────────────────────────────────────────
FILE_TYPE_LABELS = ["🖼️ 图片", "🎬 视频", "📄 PDF", "📝 Word", "📃 TXT", "📊 Excel"]

LABEL_TO_SEARCH_TYPE = {
    "🖼️ 图片": "image",
    "🎬 视频": "video",
    "📄 PDF":  "document",
    "📝 Word": "document",
    "📃 TXT":  "document",
    "📊 Excel":"document",
}


# ── 搜索函数 ────────────────────────────────────────────────
def do_search(query: str, file_types: list, top_k: int, min_score: float, hybrid_weight: float):
    """功能3+5: 文档预览 + 混合检索"""
    if not query.strip():
        return [], "请输入搜索词"
    try:
        engine = get_engine()

        # 转换文件类型
        search_types = set()
        for label in (file_types or []):
            t = LABEL_TO_SEARCH_TYPE.get(label)
            if t:
                search_types.add(t)
        if not search_types:
            search_types = None

        # 功能5: 混合检索（向量 + BM25）
        results = engine.search(
            query,
            top_k=int(top_k),
            file_types=search_types,
            min_score=min_score,
            hybrid_weight=hybrid_weight
        )

        if not results:
            return [], "❌ 未找到相关文件，请换个关键词试试"

        # 构建展示内容
        gallery_items = []
        text_lines = [f"找到 **{len(results)}** 个相关文件（混合权重={hybrid_weight}）\n"]

        for i, r in enumerate(results, 1):
            ftype = r.get("type", "unknown")
            ext = r.get("ext", "").lower()
            vec_s = r.get("vec_score", 0)
            bm25_s = r.get("bm25_score", 0)

            # 图标
            icon_map = {
                "image": "🖼️", "video": "🎬",
                ".pdf": "📄", ".docx": "📝", ".doc": "📝",
                ".txt": "📃", ".xlsx": "📊", ".xls": "📊"
            }
            icon = icon_map.get(ftype, icon_map.get(ext, "📁"))

            # 分数条
            score_bar = "█" * int(r['score'] * 20) + "░" * (20 - int(r['score'] * 20))
            vec_bar = "█" * int(vec_s * 20) + "░" * (20 - int(vec_s * 20))
            bm25_bar = "█" * int(bm25_s * 20) + "░" * (20 - int(bm25_s * 20))

            # 构建结果行
            line = (
                f"{i}. {icon} **{r['filename']}**\n"
                f"   相似度: `{r['score']:.3f}` [{score_bar}]  "
                f"VEC:`{vec_s:.3f}` [{vec_bar}]  "
                f"BM25:`{bm25_s:.3f}` [{bm25_bar}]\n"
                f"   路径: `{r['path']}`\n"
                f"   大小: {r.get('size_mb', '?')}MB | 时间: {r.get('mtime', '?')}"
            )

            # 功能3: 文档预览 - 直接展示内容摘要
            chunk_text = r.get("chunk_text", r.get("doc_text", ""))
            if chunk_text:
                preview = chunk_text[:200].replace("\n", " ").strip()
                line += f"\n   内容摘要: _{preview}_"

            text_lines.append(line)

            # 图片缩略图
            if ftype == "image" and os.path.exists(r["path"]):
                try:
                    from PIL import Image
                    img = Image.open(r["path"])
                    img.thumbnail((400, 400))
                    gallery_items.append((img, f"[{r['score']:.3f}] {r['filename']}"))
                except Exception:
                    pass

        return gallery_items, "\n\n".join(text_lines)

    except FileNotFoundError as e:
        return [], f"⚠️ {e}"
    except Exception as e:
        return [], f"❌ 搜索出错: {e}"


def do_similar_search(image_file):
    """以图搜图"""
    if image_file is None:
        return [], "请上传一张图片"
    try:
        engine = get_engine()
        img_path = image_file if isinstance(image_file, str) else image_file.name
        results = engine.search_similar(img_path, top_k=12)

        if not results:
            return [], "未找到相似图片"

        gallery_items = []
        text_lines = [f"以图搜图：找到 **{len(results)}** 个相似文件\n"]
        for i, r in enumerate(results, 1):
            if r["type"] == "image" and os.path.exists(r["path"]):
                try:
                    from PIL import Image
                    img = Image.open(r["path"])
                    img.thumbnail((400, 400))
                    gallery_items.append((img, f"[{r['score']:.3f}] {r['filename']}"))
                except Exception:
                    pass
            text_lines.append(f"{i}. [{r['score']:.3f}] {r['path']}")

        return gallery_items, "\n".join(text_lines)
    except Exception as e:
        return [], f"❌ 出错: {e}"


def get_index_stats():
    """获取索引统计（含 BM25）"""
    try:
        engine = get_engine()
        s = engine.stats()
        bm25_info = f"- BM25文档: {s.get('bm25_docs', 0)} 个" if s.get('bm25_docs', 0) > 0 else ""
        return (f"**索引统计**\n"
                f"- 总向量数：{s['total']}\n"
                f"- 图片：{s['images']}\n"
                f"- 视频：{s['videos']}\n"
                f"- 独立文档：{s.get('documents', 0)}\n"
                f"{bm25_info}\n"
                f"- 索引目录：`{s['index_dir']}`")
    except Exception as e:
        return f"⚠️ 索引未就绪：{e}\n\n请先运行 `python build_index.py --dirs 你的目录`"


def open_file_location(result_text: str):
    """从结果文本中提取并打开文件位置"""
    import re, subprocess
    paths = re.findall(r'`([A-Z]:\\[^`\n]+)`', result_text)
    if paths:
        subprocess.Popen(f'explorer /select,"{paths[0]}"')
        return f"已在资源管理器中定位: {paths[0]}"
    return "未能从结果中提取路径"


def select_all_file_types(current_selections: list):
    """全选/取消全选"""
    if len(current_selections) == len(FILE_TYPE_LABELS):
        return []
    else:
        return FILE_TYPE_LABELS


# ── Gradio UI ────────────────────────────────────────────────
with gr.Blocks(title="本地媒体语义检索系统 v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 本地图片/视频/文档语义检索系统
    
    > Chinese-CLIP + **BM25 混合检索** | GPU 加速 | 中文原生支持
    
    **新增功能**：文档内容预览 · 混合检索权重调节 · PaddleOCR 扫描 PDF 支持
    """)

    with gr.Tabs():
        # ── Tab 1: 文字搜索 ───────────────────────────────
        with gr.TabItem("🔤 文字搜索"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        placeholder="输入搜索词，如：海边日落、合同、会议纪要...",
                        label="搜索关键词（中英文均可）",
                        lines=1
                    )
                with gr.Column(scale=1):
                    search_btn = gr.Button("搜索", variant="primary", size="lg")

            # 文件类型多选
            with gr.Row():
                with gr.Column(scale=3):
                    file_type_checkboxes = gr.CheckboxGroup(
                        choices=FILE_TYPE_LABELS,
                        value=FILE_TYPE_LABELS,
                        label="文件类型（可多选）",
                        info="选择要搜索的文件类型"
                    )
                with gr.Column(scale=1):
                    select_all_btn = gr.Button("全选/取消全选", size="sm")

            with gr.Row():
                top_k = gr.Slider(5, 50, value=20, step=5, label="返回数量")
                min_score = gr.Slider(0.0, 0.5, value=0.15, step=0.05, label="最低相似度")

            # 功能5: 混合检索权重
            with gr.Row():
                hybrid_slider = gr.Slider(
                    0.0, 1.0, value=0.4, step=0.05,
                    label="BM25 权重",
                    info="0=纯向量检索，1=纯关键词检索，0.4=混合（推荐）"
                )
                gr.Markdown("""
                <details>
                <summary><b>BM25 混合检索说明</b></summary>
                
                - **VEC 分数**：向量语义相似度（理解语义，如"海滩"≈"海边"）  
                - **BM25 分数**：关键词精确匹配（"合同"必须在文档中出现）  
                - 混合权重 0.4 = **60% 向量 + 40% 关键词**，平衡语义理解和精确匹配  
                - 搜索文档时推荐 0.3~0.5，搜索图片时保持默认 0.4 即可
                </details>
                """)

            gallery_out = gr.Gallery(label="搜索结果（图片预览）", columns=4)
            text_out = gr.Markdown(label="详细结果")
            open_btn = gr.Button("在资源管理器中定位")
            open_status = gr.Textbox(label="", interactive=False)

            # 快捷搜索示例
            gr.Examples(
                examples=[
                    ["海边风景", ["🖼️ 图片"], 20, 0.15, 0.0],
                    ["小孩子", ["🖼️ 图片"], 20, 0.15, 0.0],
                    ["合同", ["📄 PDF", "📝 Word"], 20, 0.15, 0.4],
                    ["会议纪要", ["📄 PDF", "📝 Word", "📃 TXT"], 20, 0.15, 0.4],
                    ["财务报表", ["📊 Excel"], 20, 0.15, 0.4],
                ],
                inputs=[query_input, file_type_checkboxes, top_k, min_score, hybrid_slider],
                label="快捷查询"
            )

            # 事件绑定
            search_btn.click(
                do_search,
                inputs=[query_input, file_type_checkboxes, top_k, min_score, hybrid_slider],
                outputs=[gallery_out, text_out]
            )
            query_input.submit(
                do_search,
                inputs=[query_input, file_type_checkboxes, top_k, min_score, hybrid_slider],
                outputs=[gallery_out, text_out]
            )
            open_btn.click(
                open_file_location,
                inputs=[text_out],
                outputs=[open_status]
            )
            select_all_btn.click(
                lambda x: select_all_file_types(x),
                inputs=[file_type_checkboxes],
                outputs=[file_type_checkboxes]
            )

        # ── Tab 2: 以图搜图 ───────────────────────────────
        with gr.TabItem("🖼️ 以图搜图"):
            gr.Markdown("上传一张图片，找出本地库中最相似的图片")
            with gr.Row():
                upload_img = gr.Image(type="filepath", label="上传参考图片")
                similar_btn = gr.Button("找相似", variant="primary")
            similar_gallery = gr.Gallery(label="相似图片", columns=4)
            similar_text = gr.Markdown()
            similar_btn.click(
                do_similar_search,
                inputs=[upload_img],
                outputs=[similar_gallery, similar_text]
            )

        # ── Tab 3: 索引管理 ───────────────────────────────
        with gr.TabItem("⚙️ 索引管理"):
            gr.Markdown("""
            ### 如何建立/更新索引

            ```bash
            # 索引图片 + 视频 + 文档（推荐首次运行）
            python C:\\MediaSearch\\build_index.py --dirs "D:\\Photos" "E:\\Documents"

            # 仅索引文档（含分块索引 + BM25）
            python C:\\MediaSearch\\build_index.py --dirs "E:\\Docs" --types document

            # 开启图片描述（更准但慢，约 3~5 张/秒）
            python C:\\MediaSearch\\build_index.py --dirs "D:\\Photos" --caption
            ```

            ### 支持的文件格式
            
            | 类型 | 格式 | 处理方式 |
            |------|------|---------|
            | 图片 | JPG/PNG/HEIC/BMP... | Chinese-CLIP 图像编码 |
            | 视频 | MP4/AVI/MKV/MOV... | 抽帧（8帧）取平均向量 |
            | PDF | .pdf | **PaddleOCR**（文本型 + 扫描型） |
            | Word | .docx/.doc | 段落提取 |
            | TXT | .txt | 直接读取 |
            | Excel | .xlsx/.xls | 单元格提取 |
            
            ### 技术栈
            - **向量检索**：Chinese-CLIP ViT-L-14（768维，GPU加速）
            - **关键词检索**：BM25（双字词分词，中文友好）
            - **扫描 PDF**：PaddleOCR（GPU加速）
            """)
            stats_btn = gr.Button("查看索引状态", variant="secondary")
            stats_out = gr.Markdown()
            stats_btn.click(get_index_stats, outputs=[stats_out])
            demo.load(get_index_stats, outputs=[stats_out])


if __name__ == "__main__":
    print("=" * 60)
    print("  MediaSearch v2 - 混合检索版")
    print("  访问地址: http://localhost:7860")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
