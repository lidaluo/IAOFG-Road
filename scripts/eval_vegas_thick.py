"""
在拉斯维加斯厚掩膜（masks_thick）上调用 eval_vegas_aoi2 重新评估。
若 masks_thick 不存在，可先运行 scripts/preprocess_vegas_thick.py。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    p = argparse.ArgumentParser(description="Vegas AOI2 厚掩膜评估（包装 eval_vegas_aoi2）")
    p.add_argument("--vegas-root", type=str, default="E:/Code/spacenet/train/AOI2_Vegas")
    p.add_argument("--thick-masks-dir", type=str, default=None, help="默认 <vegas-root>/masks_thick")
    p.add_argument(
        "--config",
        type=str,
        default="configs/config_vegas_aoi2_eval_thick.yaml",
    )
    p.add_argument("--kernel-size", type=int, default=5, help="若需生成厚掩膜，与 preprocess 一致")
    p.add_argument("--min-road-frac", type=float, default=0.05)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--apls-max-gt-nodes", type=int, default=64)
    args = p.parse_args()

    root = args.vegas_root
    thick_dir = args.thick_masks_dir or os.path.join(root, "masks_thick")
    if not os.path.isdir(thick_dir) or not os.listdir(thick_dir):
        print(f"[INFO] 厚掩膜目录为空或不存在: {thick_dir}")
        print("[INFO] 运行 preprocess_vegas_thick …")
        prep = os.path.join(PROJECT_ROOT, "scripts", "preprocess_vegas_thick.py")
        cmd = [
            sys.executable,
            prep,
            "--input_dir",
            os.path.join(root, "masks"),
            "--output_dir",
            thick_dir,
            "--kernel_size",
            str(args.kernel_size),
        ]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    cfg = args.config
    if not os.path.isabs(cfg):
        cfg = os.path.join(PROJECT_ROOT, cfg)

    eval_script = os.path.join(PROJECT_ROOT, "scripts", "eval_vegas_aoi2.py")
    cmd = [
        sys.executable,
        eval_script,
        "--config",
        cfg,
        "--vegas-root",
        root,
        "--masks-subdir",
        "masks_thick",
        "--min-road-frac",
        str(args.min_road_frac),
        "--top-n",
        str(args.top_n),
        "--max-samples",
        str(args.max_samples),
        "--apls-max-gt-nodes",
        str(args.apls_max_gt_nodes),
    ]
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    log_eval = os.path.join(
        PROJECT_ROOT, "logs_vegas_aoi2_eval_thick", "eval", "vegas_topology_eval.json"
    )
    if os.path.isfile(log_eval):
        with open(log_eval, "r", encoding="utf-8") as f:
            r = json.load(f)
        print("\n--- Vegas thick summary ---")
        print(f"  Topology APLS: {r.get('topology_apls', 0):.4f}")
        print(f"  Pixel IoU:     {r.get('pixel_iou', 0):.4f}")
        print(f"  Pixel F1:      {r.get('pixel_f1', 0):.4f}")
        print(f"  Report:        logs_vegas_aoi2_eval_thick/eval/VEGAS_AOI2_TEST_REPORT.md")


if __name__ == "__main__":
    main()
