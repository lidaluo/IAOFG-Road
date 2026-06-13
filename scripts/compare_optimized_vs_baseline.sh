#!/usr/bin/env bash
# 基线（epoch19 后处理优化配置）vs 扩展优化最终参数；各写独立 log_dir，避免覆盖

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASELINE_CONFIG="configs/config_shanghai_thick_epoch19_postproc_opt.yaml"
OPTIMIZED_CONFIG="configs/config_shanghai_thick_optimized_final.yaml"
EVAL_SCRIPT="scripts/eval_topology.py"
OUT_BASE="eval_results"
BASELINE_JSON="${OUT_BASE}/baseline_eval.json"
OPT_JSON="${OUT_BASE}/optimized_eval.json"

mkdir -p "${OUT_BASE}"

echo "========================================="
echo "优化前后结果对比（各跑一次全量 eval）"
echo "========================================="

if [[ ! -f "${BASELINE_CONFIG}" ]]; then
  echo "缺少基线配置: ${BASELINE_CONFIG}"
  exit 1
fi
if [[ ! -f "${OPTIMIZED_CONFIG}" ]]; then
  echo "缺少优化配置: ${OPTIMIZED_CONFIG}"
  exit 1
fi

echo "1) 基线: ${BASELINE_CONFIG}"
python "${EVAL_SCRIPT}" --config "${BASELINE_CONFIG}"
cp -f logs_shanghai_thick_postproc_epoch19_best/eval/topology_eval.json "${BASELINE_JSON}"

echo "2) 优化后: ${OPTIMIZED_CONFIG}"
python "${EVAL_SCRIPT}" --config "${OPTIMIZED_CONFIG}"
cp -f logs_shanghai_thick_optimized_final/eval/topology_eval.json "${OPT_JSON}"

echo "3) 对比"
python scripts/compare_eval_json.py "${BASELINE_JSON}" "${OPT_JSON}"

echo "已保存: ${BASELINE_JSON} , ${OPT_JSON}"
