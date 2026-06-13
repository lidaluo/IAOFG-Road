import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime

import yaml


def run_cmd(command, cwd):
    print(f"[CMD] {command}")
    proc = subprocess.run(command, cwd=cwd, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def make_variants(base_cfg):
    variants = []
    # 1) Seg-only baseline
    v = copy.deepcopy(base_cfg)
    v["training"]["lambda_inter"] = 0.0
    v["training"]["lambda_orient"] = 0.0
    v["training"]["lambda_anchor"] = 0.0
    variants.append(("seg_only", v))

    # 2) +Intersection
    v = copy.deepcopy(base_cfg)
    v["training"]["lambda_inter"] = 0.3
    v["training"]["lambda_orient"] = 0.0
    v["training"]["lambda_anchor"] = 0.0
    variants.append(("seg_plus_inter", v))

    # 3) +Orientation
    v = copy.deepcopy(base_cfg)
    v["training"]["lambda_inter"] = 0.0
    v["training"]["lambda_orient"] = 0.3
    v["training"]["lambda_anchor"] = 0.0
    variants.append(("seg_plus_orient", v))

    # 4) IAOF full
    v = copy.deepcopy(base_cfg)
    v["training"]["lambda_inter"] = 0.3
    v["training"]["lambda_orient"] = 0.3
    v["training"]["lambda_anchor"] = 0.1
    variants.append(("iaof_full", v))

    # 5) IAOF + stronger topo
    v = copy.deepcopy(base_cfg)
    v["training"]["lambda_inter"] = 0.3
    v["training"]["lambda_orient"] = 0.3
    v["training"]["lambda_anchor"] = 0.15
    variants.append(("iaof_full_anchor_strong", v))

    return variants


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiment matrix.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--run-train", action="store_true", help="Actually run training.")
    parser.add_argument("--run-eval", action="store_true", help="Run eval_topology after training.")
    args = parser.parse_args()

    with open(os.path.join(args.workspace, args.config), "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.workspace, "experiments", f"ablation_{ts}")
    os.makedirs(out_root, exist_ok=True)

    variants = make_variants(base_cfg)
    summary = []
    for name, cfg in variants:
        run_dir = os.path.join(out_root, name)
        os.makedirs(run_dir, exist_ok=True)
        cfg["training"]["checkpoint_dir"] = os.path.join(run_dir, "checkpoints")
        cfg["training"]["log_dir"] = os.path.join(run_dir, "logs")
        cfg_path = os.path.join(run_dir, "config.yaml")
        write_yaml(cfg_path, cfg)

        item = {"variant": name, "config": cfg_path}
        if args.run_train:
            run_cmd(f"\"{args.python}\" scripts/train.py --config \"{cfg_path}\"", cwd=args.workspace)
        if args.run_eval:
            run_cmd(f"\"{args.python}\" scripts/eval_topology.py --config \"{cfg_path}\"", cwd=args.workspace)
        summary.append(item)

    with open(os.path.join(out_root, "matrix_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Ablation] manifest saved: {os.path.join(out_root, 'matrix_manifest.json')}")


if __name__ == "__main__":
    main()

