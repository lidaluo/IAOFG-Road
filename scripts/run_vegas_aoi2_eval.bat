@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo Vegas AOI2 泛化评估（上海 thick 权重）
python scripts\eval_vegas_aoi2.py --config configs\config_vegas_aoi2_eval.yaml --min-road-frac 0.05 --max-samples 0
if errorlevel 1 exit /b 1
echo.
echo 报告: logs_vegas_aoi2_eval\eval\VEGAS_AOI2_TEST_REPORT.md
echo 预测: E:\Code\spacenet\train\AOI2_Vegas\predictions  （或 config 中 root\predictions）
pause
