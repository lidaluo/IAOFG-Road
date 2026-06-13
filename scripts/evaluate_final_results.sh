#!/usr/bin/env bash
# 论文最终结果评估：与 eval_topology.py 实际 CLI 一致（仅 --config，可选覆盖后处理）

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/config_shanghai_thick_optimized_final.yaml"
CHECKPOINT="checkpoints_shanghai_thick/model_best_val_iou.pth"
EVAL_SCRIPT="scripts/eval_topology.py"
OUTPUT_DIR="eval_results/final_optimized"
LOG_FILE="${OUTPUT_DIR}/evaluation_log.txt"
RESULTS_SRC="logs_shanghai_thick_optimized_final/eval/topology_eval.json"
RESULTS_FILE="${OUTPUT_DIR}/final_results.json"

mkdir -p "${OUTPUT_DIR}"

start_ts=$(date +%s)
echo "========================================="
echo "最终优化结果评估"
echo "========================================="
{
  echo "评估开始: $(date)"
  echo "模型: ${CHECKPOINT}"
  echo "配置: ${CONFIG}"
  echo "========================================"
  echo
} | tee "${LOG_FILE}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "检查点不存在: ${CHECKPOINT}" | tee -a "${LOG_FILE}"
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "配置文件不存在: ${CONFIG}" | tee -a "${LOG_FILE}"
  exit 1
fi

echo "运行最终评估（全量验证集）..." | tee -a "${LOG_FILE}"
echo "这可能需要较长时间..." | tee -a "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"

if python "${EVAL_SCRIPT}" --config "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"; then
  echo "评估完成" | tee -a "${LOG_FILE}"
else
  echo "评估失败" | tee -a "${LOG_FILE}"
  exit 1
fi

if [[ -f "${RESULTS_SRC}" ]]; then
  cp -f "${RESULTS_SRC}" "${RESULTS_FILE}"
  echo | tee -a "${LOG_FILE}"
  python scripts/print_eval_summary.py "${RESULTS_FILE}" | tee -a "${LOG_FILE}"
else
  echo "未找到结果: ${RESULTS_SRC}" | tee -a "${LOG_FILE}"
  exit 1
fi

end_ts=$(date +%s)
dur=$((end_ts - start_ts))
min=$((dur / 60))
sec=$((dur % 60))
echo | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "结束时间: $(date)" | tee -a "${LOG_FILE}"
echo "总耗时: ${min}分${sec}秒" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "结果 JSON: ${RESULTS_FILE}"
echo "详细日志: ${LOG_FILE}"
echo "原始 eval 输出: ${RESULTS_SRC}"
