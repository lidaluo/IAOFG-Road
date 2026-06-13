@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."

set CONFIG=configs\config_shanghai_thick_optimized_final.yaml
set CHECKPOINT=checkpoints_shanghai_thick\model_best_val_iou.pth
set EVAL_SCRIPT=scripts\eval_topology.py
set OUTPUT_DIR=eval_results\final_optimized
set LOG_FILE=%OUTPUT_DIR%\evaluation_log.txt
set RESULTS_SRC=logs_shanghai_thick_optimized_final\eval\topology_eval.json
set RESULTS_FILE=%OUTPUT_DIR%\final_results.json

if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

echo =========================================
echo 最终优化结果评估
echo =========================================
echo 评估开始: %date% %time% > "%LOG_FILE%"
echo 模型: %CHECKPOINT% >> "%LOG_FILE%"
echo 配置: %CONFIG% >> "%LOG_FILE%"
echo ======================================== >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

if not exist %CHECKPOINT% (
    echo 检查点不存在: %CHECKPOINT% >> "%LOG_FILE%"
    echo 检查点不存在: %CHECKPOINT%
    pause
    exit /b 1
)
if not exist %CONFIG% (
    echo 配置文件不存在: %CONFIG% >> "%LOG_FILE%"
    echo 配置文件不存在: %CONFIG%
    pause
    exit /b 1
)

echo 运行最终评估（全量验证集）... >> "%LOG_FILE%"
echo 这可能需要较长时间... >> "%LOG_FILE%"
echo.
echo 运行最终评估（全量验证集）...
echo 这可能需要较长时间...
echo.

python %EVAL_SCRIPT% --config %CONFIG% >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo 评估失败 >> "%LOG_FILE%"
    echo 评估失败
    pause
    exit /b 1
)
echo 评估完成 >> "%LOG_FILE%"
echo 评估完成

if exist %RESULTS_SRC% (
    copy /Y %RESULTS_SRC% %RESULTS_FILE% >nul
    echo. >> "%LOG_FILE%"
    python scripts\print_eval_summary.py %RESULTS_FILE% >> "%LOG_FILE%"
    echo.
    python scripts\print_eval_summary.py %RESULTS_FILE%
) else (
    echo 未找到结果: %RESULTS_SRC% >> "%LOG_FILE%"
    echo 未找到结果: %RESULTS_SRC%
    pause
    exit /b 1
)

echo.
echo =========================================
echo 结果 JSON: %RESULTS_FILE%
echo 详细日志: %LOG_FILE%
echo 原始 eval 输出: %RESULTS_SRC%
echo =========================================
pause
