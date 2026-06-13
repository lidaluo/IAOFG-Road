#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VEGAS_ROOT="${VEGAS_ROOT:-E:/Code/spacenet/train/AOI2_Vegas}"

echo "========================================="
echo "Vegas AOI2 厚掩膜 + 评估"
echo "========================================="

echo "[1/2] 生成厚掩膜"
python scripts/preprocess_vegas_thick.py \
  --input_dir "${VEGAS_ROOT}/masks" \
  --output_dir "${VEGAS_ROOT}/masks_thick" \
  --kernel_size 5 \
  --visualize

echo
echo "[2/2] 厚掩膜上评估"
python scripts/eval_vegas_thick.py \
  --vegas-root "${VEGAS_ROOT}" \
  --config configs/config_vegas_aoi2_eval_thick.yaml \
  --min-road-frac 0.05 \
  --top-n 20 \
  --max-samples 0

echo
echo "完成。报告: logs_vegas_aoi2_eval_thick/eval/VEGAS_AOI2_TEST_REPORT.md"
