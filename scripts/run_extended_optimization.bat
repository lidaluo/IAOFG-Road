@echo off
chcp 65001 >nul
setlocal

set CONFIG=configs\config_shanghai_thick.yaml
set CHECKPOINT=checkpoints_shanghai_thick\model_best_val_iou.pth
set OPTIMIZER_SCRIPT=scripts\optimize_postprocessing_extended.py
set OUTPUT_DIR=eval_results\extended_optimization_v2
set LOG_FILE=%OUTPUT_DIR%\run_log.txt
set TIME_LIMIT=7200
set COARSE_N=64

if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

echo ========================================= > %LOG_FILE%
echo 扩展后处理优化（子集粗搜+全量复核） >> %LOG_FILE%
echo ========================================= >> %LOG_FILE%

if not exist %CHECKPOINT% (
    set CHECKPOINT=checkpoints_shanghai_thick\checkpoint_epoch_19.pth
)

if not exist %CHECKPOINT% (
    echo 检查点不存在 >> %LOG_FILE%
    echo 请指定 checkpoints_shanghai_thick\model_best_val_iou.pth
    exit /b 1
)

if not exist %CONFIG% (
    echo 配置文件不存在: %CONFIG% >> %LOG_FILE%
    exit /b 1
)

echo 开始优化 coarse_num_samples=%COARSE_N% ... >> %LOG_FILE%
python %OPTIMIZER_SCRIPT% ^
  --config %CONFIG% ^
  --checkpoint %CHECKPOINT% ^
  --output_dir %OUTPUT_DIR% ^
  --coarse_num_samples %COARSE_N% ^
  --grid fast ^
  --time_limit %TIME_LIMIT% >> %LOG_FILE% 2>&1

echo 完成: %OUTPUT_DIR% >> %LOG_FILE%
echo 完成: %OUTPUT_DIR%
endlocal
