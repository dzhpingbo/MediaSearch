@echo off
chcp 65001 > nul
echo ============================================
echo  MediaSearch 全量索引
echo ============================================
echo.

REM 清除代理，防止下载失败
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set no_proxy=*
set NO_PROXY=*

cd /d C:\MediaSearch

echo 开始索引 1588 个文件，预计需要 20~30 分钟...
echo 日志将写入 index_full_log.txt
echo.

"C:\Users\Administrator\.conda\envs\aimodel\python.exe" -u build_index.py 2>&1 | powershell -Command "$input | Tee-Object -FilePath index_full_log.txt"

echo.
echo 索引完成！
pause
