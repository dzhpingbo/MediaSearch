"""
app.py - 本地图片/视频/文档语义检索 Web UI（Chinese-CLIP 版）
运行: python app.py
然后浏览器访问 http://localhost:7860
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
# CheckboxGroup 的 choices 直接用中文 label（Gradio 传回 value = label 本身）
FILE_TYPE_LABELS = ["🖼️ 图片", "🎬 视频", "📄 PDF", "📝 Word", "📃 TXT", "📊 Excel"]

# label → search.py 期望的 type 字符串（image/video/document）
LABEL_TO_SEARCH_TYPE = {
    "🖼️ 图片": "image",
    "🎬 视频": "video",
    "📄 PDF":  "document",
    "📝 Word": "document",
    "📃 TXT":  "document",
    "📊 Excel":"document",
}


# ── 搜索函数 ────────────────────────────────────────────────
def do_search(query: str, file_types: list, top_k: int, min_score: float):
    if not query.strip():
        return [], "请输入搜索词"
    try:
        engine = get_engine()

        # 将 UI 选择的 label 列表转换为 search.py 期望的 type 集合
        search_types = set()
        for label in (file_types or []):
            t = LABEL_TO_SEARCH_TYPE.get(label)
            if t:
                search_types.add(t)

        if not search_types:
            search_types = None  # 全选或空 = 搜索所有类型

        results = engine.search(query, top_k=int(top_k), file_types=search_types, min_score=min_score)

        if not results:
            return [], "❌ 未找到相关文件，请换个关键词试试"

        # 构建展示内容
        gallery_items = []
        text_lines = [f"找到 **{len(results)}** 个相关文件\n"]

        for i, r in enumerate(results, 1):
            ftype = r.get("type", "unknown")
            ext = r.get("ext", "").lower()

            # 选择图标
            if ftype == "image":
                icon = "🖼️"
            elif ftype == "video":
                icon = "🎬"
            else:
                if ext == ".pdf":
                    icon = "📄"
                elif ext in [".docx", ".doc"]:
                    icon = "📝"
                elif ext == ".txt":
                    icon = "📃"
                elif ext in [".xlsx", ".xls"]:
                    icon = "📊"
                else:
                    icon = "📁"

            score_bar = "█" * int(r['score'] * 20) + "░" * (20 - int(r['score'] * 20))
            line = (f"{i}. {icon} `{r['filename']}` | 相似度: {r['score']:.3f} [{score_bar}]\n"
                    f"   📁 `{r['path']}`\n"
                    f"   💾 {r.get('size_mb', '?')}MB | 📅 {r.get('mtime', '?')}")
            caption = r.get("caption", "")
            if caption:
                line += f"\n   💬 {caption}"
            text_lines.append(line)

            # 图片类型直接展示缩略图
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
        # gradio 上传的文件路径
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
    """获取索引统计"""
    try:
        engine = get_engine()
        s = engine.stats()
        return (f"📊 **索引统计**\n"
                f"- 总文件数：{s['total']}\n"
                f"- 图片：{s['images']}\n"
                f"- 视频：{s['videos']}\n"
                f"- 文档：{s.get('documents', 0)}\n"
                f"- 索引目录：`{s['index_dir']}`")
    except Exception as e:
        return f"⚠️ 索引未就绪：{e}\n\n请先运行 `python build_index.py --dirs 你的目录`"


def open_file_location(result_text: str):
    """从结果文本中提取并打开文件位置"""
    import re, subprocess
    paths = re.findall(r'`([A-Z]:\\[^`\n]+)`', result_text)
    if paths:
        # 打开第一个文件所在文件夹
        folder = str(Path(paths[0]).parent)
        subprocess.Popen(f'explorer /select,"{paths[0]}"')
        return f"✅ 已在资源管理器中定位: {paths[0]}"
    return "❌ 未能从结果中提取路径"


def select_all_file_types(current_selections: list):
    """全选/取消全选逻辑"""
    if len(current_selections) == len(FILE_TYPE_LABELS):
        return []  # 已全选 → 取消全选
    else:
        return FILE_TYPE_LABELS  # 否则全选


# ── Gradio UI ────────────────────────────────────────────────
with gr.Blocks(title="本地媒体语义检索系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 本地图片/视频/文档语义检索系统
    > 基于 Chinese-CLIP + GPU 加速 | 支持中英文自然语言搜索
    """)

    with gr.Tabs():
        # ── Tab 1: 文字搜索 ──────────────────────────────────
        with gr.TabItem("🔤 文字搜索"):
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        placeholder="输入搜索词，如：海边日落、孩子玩耍、合同、会议纪要...",
                        label="搜索关键词（支持中英文）",
                        lines=1
                    )
                with gr.Column(scale=1):
                    search_btn = gr.Button("🔍 搜索", variant="primary", size="lg")

            # 文件类型多选
            with gr.Row():
                with gr.Column(scale=3):
                    file_type_checkboxes = gr.CheckboxGroup(
                        choices=FILE_TYPE_LABELS,
                        value=FILE_TYPE_LABELS,  # 默认全选
                        label="文件类型（可多选）",
                        info="选择要搜索的文件类型"
                    )
                with gr.Column(scale=1):
                    select_all_btn = gr.Button("全选/取消全选", size="sm")

            with gr.Row():
                top_k = gr.Slider(5, 50, value=20, step=5, label="返回数量")
                min_score = gr.Slider(0.0, 0.5, value=0.15, step=0.05, label="最低相似度")

            gallery_out = gr.Gallery(label="搜索结果（图片预览）",
                                     columns=4, height=400)
            text_out = gr.Markdown(label="详细结果")
            open_btn = gr.Button("📂 在资源管理器中定位第一个结果")
            open_status = gr.Textbox(label="", interactive=False)

            # 快捷搜索示例
            gr.Examples(
                examples=[
                    ["海边风景", ["🖼️ 图片"], 20],
                    ["小孩子", ["🖼️ 图片"], 20],
                    ["合同", ["📄 PDF", "📝 Word"], 20],
                    ["会议纪要", ["📄 PDF", "📝 Word", "📃 TXT"], 20],
                    ["财务报表", ["📊 Excel"], 20],
                ],
                inputs=[query_input, file_type_checkboxes, top_k],
                label="示例查询"
            )

            # 事件绑定
            search_btn.click(
                do_search,
                inputs=[query_input, file_type_checkboxes, top_k, min_score],
                outputs=[gallery_out, text_out]
            )
            query_input.submit(
                do_search,
                inputs=[query_input, file_type_checkboxes, top_k, min_score],
                outputs=[gallery_out, text_out]
            )
            open_btn.click(
                open_file_location,
                inputs=[text_out],
                outputs=[open_status]
            )

            # 全选/取消全选
            select_all_btn.click(
                lambda x: select_all_file_types(x),
                inputs=[file_type_checkboxes],
                outputs=[file_type_checkboxes]
            )

        # ── Tab 2: 以图搜图 ──────────────────────────────────
        with gr.TabItem("🖼️ 以图搜图"):
            gr.Markdown("上传一张图片，找出本地库中最相似的图片")
            with gr.Row():
                upload_img = gr.Image(type="filepath", label="上传参考图片")
                similar_btn = gr.Button("🔍 找相似", variant="primary")
            similar_gallery = gr.Gallery(label="相似图片", columns=4, height=400)
            similar_text = gr.Markdown()
            similar_btn.click(
                do_similar_search,
                inputs=[upload_img],
                outputs=[similar_gallery, similar_text]
            )

        # ── Tab 3: 索引管理 ──────────────────────────────────
        with gr.TabItem("⚙️ 索引管理"):
            gr.Markdown("""
            ### 如何建立索引

            在命令行运行（替换路径为你的实际图片/视频/文档目录）：

            ```bash
            # 基础版（仅向量索引，速度快）
            python C:\\MediaSearch\\build_index.py --dirs "D:\\Photos" "E:\\Documents"

            # 增强版（同时生成图片描述，更准确，但更慢）
            python C:\\MediaSearch\\build_index.py --dirs "D:\\Photos" --caption

            # 仅索引特定类型
            python C:\\MediaSearch\\build_index.py --dirs "E:\\Docs" --types document
            ```

            ### 支持的文件格式
            - **图片**：JPG、PNG、BMP、GIF、WEBP、TIFF、HEIC
            - **视频**：MP4、AVI、MKV、MOV、WMV、FLV、M4V
            - **文档**：PDF、DOCX、DOC、TXT、XLSX、XLS

            ### 性能参考（GTX 1080 Ti）
            - 图片：约 **10~20 张/秒**
            - 视频：约 **2~5 个/秒**（需要抽帧）
            - 文档：约 **5~10 个/秒**（需要提取文本）
            """)

            stats_btn = gr.Button("📊 查看索引状态", variant="secondary")
            stats_out = gr.Markdown()
            stats_btn.click(get_index_stats, outputs=[stats_out])

            # 自动加载统计
            demo.load(get_index_stats, outputs=[stats_out])


if __name__ == "__main__":
    print("=" * 60)
    print("  本地媒体语义检索系统")
    print("  访问地址: http://localhost:7860")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
