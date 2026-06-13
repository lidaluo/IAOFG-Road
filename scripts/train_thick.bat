@echo off
cd /d "%~dp0.."
echo ========================================
echo Thick dataset training
echo ========================================

echo Step 1: verify dataset...
set PY=D:\ProgramFlies\Anaconda3\envs\road_extraction\python.exe
if not exist "%PY%" set PY=python
"%PY%" scripts\verify_thick_dataset.py
if errorlevel 1 (
  echo Verify failed or no samples — check spacenet_filtered_thick
  pause
  exit /b 1
)

echo.
echo Step 2: train (config_shanghai_thick.yaml^)
echo Logs: logs_shanghai_thick
echo.
"%PY%" scripts\train.py --config configs\config_shanghai_thick.yaml

echo.
echo Done. See logs_shanghai_thick\training_log.json
pause
