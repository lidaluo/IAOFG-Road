@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."

set EVAL_DIR=eval_results\final_optimized
set PAPER_DIR=paper_materials
set LOG_FILE=%PAPER_DIR%\preparation_log.txt

if not exist %PAPER_DIR% mkdir %PAPER_DIR%

echo =========================================
echo 开始准备论文材料
echo =========================================
echo.

echo 开始时间: %date% %time% > "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo 1. 整理方法和实验结果... >> "%LOG_FILE%"
python scripts\prepare_paper_materials.py --eval_dir %EVAL_DIR% --num_samples 6 >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo 材料整理失败，详见日志: %LOG_FILE%
    pause
    exit /b 1
)

echo 2. 生成拓扑可视化图... >> "%LOG_FILE%"
python -m pip install matplotlib -q >> "%LOG_FILE%" 2>&1
python scripts\generate_topology_figures.py --num_samples 4 --data_dir %PAPER_DIR% >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo 可视化生成失败，详见日志: %LOG_FILE%
    pause
    exit /b 1
)

echo ========================================= >> "%LOG_FILE%"
echo 论文材料准备完成 >> "%LOG_FILE%"
echo 结束时间: %date% %time% >> "%LOG_FILE%"
echo ========================================= >> "%LOG_FILE%"

echo 论文材料准备完成!
echo 输出目录: %PAPER_DIR%\
echo 日志文件: %LOG_FILE%
pause
