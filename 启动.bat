@echo off
chcp 65001 >nul
title 本地媒体语义检索系统 (Chinese-CLIP)

REM ── 禁用代理 ──
set http_proxy=
set https_proxy=
set HTTP_PROXY=
set HTTPS_PROXY=

REM ── 强制使用 aimodel conda 环境的 Python（torch 2.0.1+cu117 / GTX 1080 Ti）
set PYTHON=C:\Users\Administrator\.conda\envs\aimodel\python.exe
set SCRIPTS=C:\MediaSearch

echo ================================================
echo   本地图片/视频语义检索系统 - 启动器
echo   模型: Chinese-CLIP ViT-L-14
echo   环境: aimodel (torch 2.0.1+cu117 / CUDA 11.7)
echo ================================================
echo.

REM 验证环境
%PYTHON% -c "import torch; assert '2.0' in torch.__version__ and 'cu117' in torch.__version__; print('[OK] torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [错误] torch 环境异常，请检查 conda aimodel 环境
    pause
    exit /b 1
)

REM 检查是否传入扫描目录参数
if "%~1"=="" (
    echo 未传入扫描目录，直接启动 Web UI...
    goto launch_ui
)

echo [步骤1] 构建图片/视频向量索引...
echo 扫描目录: %*
echo.
%PYTHON% %SCRIPTS%\build_index.py --dirs %* --output C:\MediaSearch\index
if errorlevel 1 (
    echo 索引构建失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo [步骤2] 索引构建完成！启动 Web 检索界面...

:launch_ui
echo.
echo 启动 Web UI -> http://localhost:7860
echo 按 Ctrl+C 关闭服务
echo.
%PYTHON% %SCRIPTS%\app.py

pause
