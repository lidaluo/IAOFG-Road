#!/bin/bash
# 扩展后处理优化：子集 64 + fast 网格 + 全量复核

set -e
CONFIG="configs/config_shanghai_thick.yaml"
CHECKPOINT="checkpoints_shanghai_thick/model_best_val_iou.pth"
if [ ! -f "$CHECKPOINT" ]; then
  CHECKPOINT="checkpoints_shanghai_thick/checkpoint_epoch_19.pth"
fi
OPTIMIZER_SCRIPT="scripts/optimize_postprocessing_extended.py"
OUTPUT_DIR="eval_results/extended_optimization_v2"
LOG_FILE="${OUTPUT_DIR}/run_log.txt"
TIME_LIMIT=7200
COARSE_N=64

mkdir -p "${OUTPUT_DIR}"

{
  echo "========================================="
  echo "扩展后处理优化（子集粗搜+全量复核）"
  echo "========================================="
  python "${OPTIMIZER_SCRIPT}" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --coarse_num_samples "${COARSE_N}" \
    --grid fast \
    --time_limit "${TIME_LIMIT}"
} 2>&1 | tee -a "${LOG_FILE}"

echo "完成: ${OUTPUT_DIR}"
