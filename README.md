# MediaSearch - 本地图片/视频/文档语义检索系统

> **模型**：Chinese-CLIP ViT-L-14 | 原生中文理解 | GPU 加速 | GTX 1080 Ti  
> **版本**：v1.0.0 | 2026-05-05

---

## 一句话理解

**用中文描述要找的图片/视频/文档，系统自动找到最相似的文件。**

---

## 功能特性

- **自然语言搜索**：输入"海边日落的照片"、"会议纪要"、"合同文件"，中英文均可
- **多格式支持**：图片（JPG/PNG/HEIC...）、视频（MP4/AVI/MKV...）、PDF、Word、TXT、Excel
- **以图搜图**：上传一张参考图，找出本地最相似的图片
- **GPU 加速**：GTX 1080 Ti，每秒处理 15~25 张图片
- **断点续传**：索引中途断电/中断，重新运行自动跳过已处理文件
- **Web 界面**：无需命令行，浏览器即可操作

---

## 文件结构

```
C:\MediaSearch\
├── core_encoder.py     # 编码器（图片/视频/文档 → 向量）
├── build_index.py      # 索引构建脚本
├── search.py           # 检索引擎
├── app.py              # Gradio Web 界面
├── requirements.txt    # 依赖清单
├── .gitignore
├── README.md
└── index/              # 向量索引（自动生成，不上传 Git）
    ├── vectors.faiss   # FAISS 向量文件
    └── metadata.json   # 文件元数据
```

---

## 快速开始

### 环境要求

- **GPU**：NVIDIA GTX 1080 Ti（11GB）或更强
- **系统**：Windows（已测试）| macOS/Linux 通用
- **Python**：3.11.4（建议 conda 环境）
- **CUDA**：11.7

### 安装依赖

```bash
# 1. 创建 conda 环境（已有 aimodel 环境可跳过）
conda create -n mediasearch python=3.11.4
conda activate mediasearch

# 2. 安装 PyTorch（CUDA 11.7）
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117

# 3. 安装本项目依赖
pip install -r requirements.txt
```

### 建立索引

```bash
# 索引图片和视频
python build_index.py --dirs "D:\Photos" "E:\Videos"

# 索引文档（PDF/Word/TXT/Excel）
python build_index.py --dirs "C:\Documents" --types document

# 索引全部类型
python build_index.py --dirs "D:\Photos" "E:\Documents" --types image,video,document

# 开启图片描述（更准但更慢，约 3~5 张/秒）
python build_index.py --dirs "D:\Photos" --caption
```

### 启动 Web 界面

```bash
python app.py
# 浏览器访问 http://localhost:7860
```

### 命令行搜索

```bash
# 搜索图片
python search.py "孩子玩耍" --types image --top 10

# 搜索文档
python search.py "合同" --types document

# 搜索视频
python search.py "会议" --types video
```

---

## 技术架构

```
用户输入（中文文字 / 上传图片）
        ↓
   Gradio Web UI  (app.py, http://localhost:7860)
        ↓
   检索引擎  (search.py)
        ↓              ↓
   FAISS 向量索引      元数据
   (vectors.faiss)     (metadata.json)
        ↓
   编码器  (core_encoder.py)
        ↓
   Chinese-CLIP ViT-L-14  ←  GPU 加速
   768 维向量，余弦相似度
```

---

## 支持的文件格式

| 类型 | 扩展名 | 处理方式 |
|------|--------|---------|
| 图片 | jpg/jpeg/png/bmp/gif/webp/tiff/heic/heif | Chinese-CLIP 图像编码 |
| 视频 | mp4/avi/mkv/mov/wmv/flv/m4v/ts/rmvb | 均匀抽帧（默认8帧）取平均向量 |
| PDF | .pdf | pdfplumber 提取文本 → 文本编码 |
| Word | .docx/.doc | python-docx 提取段落 → 文本编码 |
| TXT | .txt | 直接读取（支持 utf-8/gbk/gb2312） |
| Excel | .xlsx/.xls | openpyxl 提取单元格 → 文本编码 |

> **注意**：扫描型 PDF（图片转成的 PDF）暂不支持，需要 OCR 支持。

---

## 性能参考（GTX 1080 Ti）

| 操作 | 速度 |
|------|------|
| 图片编码（无描述） | 15~25 张/秒 |
| 图片编码（开启描述） | 3~5 张/秒 |
| 视频编码 | 2~5 个/秒 |
| 文档编码 | 5~10 个/秒 |
| 语义检索（18646条） | <1 秒 |

---

## 常见问题

**Q: 之前用其他 CLIP 模型建的索引还能用吗？**
A: 不能，需要用 Chinese-CLIP 重新编码重建索引。

**Q: 显存不够？**
A: 在 `core_encoder.py` 中将 `_CN_CLIP_MODEL = "ViT-L-14"` 改为 `"ViT-B-16"`（更快但精度略降）。

**Q: 索引目录在哪修改？**
A: `build_index.py` 第 31 行 `DEFAULT_OUTPUT`；`app.py` 第 20 行 `INDEX_DIR`。

**Q: 启动时报代理错误？**
A: 已在代码中自动禁用代理。如仍有问题，在系统网络设置中临时关闭代理。

---

## 版本历史

- **v1.0.0** (2026-05-05)：初始版本，支持图片/视频/PDF/Word/TXT/Excel 检索，Gradio 6.x Web 界面
