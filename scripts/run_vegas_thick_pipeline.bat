@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."

set VEGAS_ROOT=E:\Code\spacenet\train\AOI2_Vegas

echo =========================================
echo Vegas AOI2 厚掩膜 + 评估
echo =========================================
echo.

echo [1/2] 生成厚掩膜
python scripts\preprocess_vegas_thick.py ^
  --input_dir "%VEGAS_ROOT%\masks" ^
  --output_dir "%VEGAS_ROOT%\masks_thick" ^
  --kernel_size 5 ^
  --visualize

if errorlevel 1 (
    echo 预处理失败
    pause
    exit /b 1
)

echo.
echo [2/2] 厚掩膜上评估（与上海 thick 权重一致）
python scripts\eval_vegas_thick.py ^
  --vegas-root "%VEGAS_ROOT%" ^
  --config configs\config_vegas_aoi2_eval_thick.yaml ^
  --min-road-frac 0.05 ^
  --top-n 20 ^
  --max-samples 0

if errorlevel 1 (
    echo 评估失败
    pause
    exit /b 1
)

echo.
echo =========================================
echo 完成
echo 厚掩膜: %VEGAS_ROOT%\masks_thick\
echo 预测:   %VEGAS_ROOT%\predictions_thick\
echo 报告:   logs_vegas_aoi2_eval_thick\eval\VEGAS_AOI2_TEST_REPORT.md
echo 对比图: %VEGAS_ROOT%\thick_mask_comparison.png
echo =========================================
pause
