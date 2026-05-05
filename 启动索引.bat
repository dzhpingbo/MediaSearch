@echo off
REM 启动 MediaSearch 全量索引
REM 索引输出目录: E:\dzhwork\pictureVedioIndex

set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set no_proxy=*
set NO_PROXY=*

cd /d C:\MediaSearch
"C:\Users\Administrator\.conda\envs\aimodel\python.exe" -u build_index.py > index_full_log.txt 2>&1
pause
