#!/usr/bin/env bash
# 论文材料准备脚本

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EVAL_DIR="eval_results/final_optimized"
PAPER_DIR="paper_materials"
LOG_FILE="${PAPER_DIR}/preparation_log.txt"

mkdir -p "${PAPER_DIR}"

echo "========================================="
echo "开始准备论文材料"
echo "========================================="
echo

start_time="$(date)"
echo "开始时间: ${start_time}" | tee "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"

echo "1. 整理方法和实验结果..." | tee -a "${LOG_FILE}"
python scripts/prepare_paper_materials.py --eval_dir "${EVAL_DIR}" --num_samples 6 2>&1 | tee -a "${LOG_FILE}"

echo | tee -a "${LOG_FILE}"
echo "2. 生成拓扑可视化图..." | tee -a "${LOG_FILE}"
python -m pip install matplotlib -q 2>&1 | tee -a "${LOG_FILE}"
python scripts/generate_topology_figures.py --num_samples 4 --data_dir "${PAPER_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo "论文材料准备完成!" | tee -a "${LOG_FILE}"
echo "结束时间: $(date)" | tee -a "${LOG_FILE}"
echo "=========================================" | tee -a "${LOG_FILE}"
echo
echo "输出目录: ${PAPER_DIR}/"
echo "日志文件: ${LOG_FILE}"
