@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."

set BASELINE_CONFIG=configs\config_shanghai_thick_epoch19_postproc_opt.yaml
set OPTIMIZED_CONFIG=configs\config_shanghai_thick_optimized_final.yaml
set EVAL_SCRIPT=scripts\eval_topology.py
set OUT_BASE=eval_results
set BASELINE_JSON=%OUT_BASE%\baseline_eval.json
set OPT_JSON=%OUT_BASE%\optimized_eval.json

if not exist %OUT_BASE% mkdir %OUT_BASE%

echo =========================================
echo 优化前后结果对比（各跑一次全量 eval）
echo =========================================

if not exist %BASELINE_CONFIG% (
    echo 缺少基线配置: %BASELINE_CONFIG%
    pause
    exit /b 1
)
if not exist %OPTIMIZED_CONFIG% (
    echo 缺少优化配置: %OPTIMIZED_CONFIG%
    pause
    exit /b 1
)

echo 1^) 基线: %BASELINE_CONFIG%
python %EVAL_SCRIPT% --config %BASELINE_CONFIG%
if errorlevel 1 (
    echo 基线评估失败
    pause
    exit /b 1
)
copy /Y logs_shanghai_thick_postproc_epoch19_best\eval\topology_eval.json %BASELINE_JSON%

echo 2^) 优化后: %OPTIMIZED_CONFIG%
python %EVAL_SCRIPT% --config %OPTIMIZED_CONFIG%
if errorlevel 1 (
    echo 优化评估失败
    pause
    exit /b 1
)
copy /Y logs_shanghai_thick_optimized_final\eval\topology_eval.json %OPT_JSON%

echo 3^) 对比
python scripts\compare_eval_json.py %BASELINE_JSON% %OPT_JSON%

echo 已保存: %BASELINE_JSON% , %OPT_JSON%
pause
