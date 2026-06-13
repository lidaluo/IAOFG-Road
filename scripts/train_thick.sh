#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "========================================"
echo "Thick dataset training"
echo "========================================"

echo "Step 1: verify dataset..."
python scripts/verify_thick_dataset.py

echo ""
echo "Step 2: train..."
python scripts/train.py --config configs/config_shanghai_thick.yaml

echo ""
echo "Done. See logs_shanghai_thick/training_log.json"
