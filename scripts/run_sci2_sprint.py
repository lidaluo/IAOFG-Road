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


def build_profiles(base_cfg):
    profiles = {}

    # Profile A: 保守稳定，先让分割脱离常数图
    a = copy.deepcopy(base_cfg)
    a["training"]["learning_rate"] = 5e-5
    a["training"]["lambda_inter"] = 0.2
    a["training"]["lambda_orient"] = 0.2
    a["training"]["lambda_anchor"] = 0.05
    a["training"]["seg_warmup_epochs"] = 6
    a["training"]["multitask_ramp_epochs"] = 6
    a["training"]["seg_pos_weight"] = 40.0
    a["training"]["inter_pos_weight"] = 30.0
    a["training"]["use_dynamic_pos_weight"] = False
    a["training"]["use_dynamic_seg_pos_weight"] = False
    a["training"]["use_dynamic_inter_pos_weight"] = False
    profiles["profile_a_stable"] = a

    # Profile B: 轻动态，兼顾召回
    b = copy.deepcopy(base_cfg)
    b["training"]["learning_rate"] = 1e-4
    b["training"]["lambda_inter"] = 0.25
    b["training"]["lambda_orient"] = 0.25
    b["training"]["lambda_anchor"] = 0.08
    b["training"]["seg_warmup_epochs"] = 5
    b["training"]["multitask_ramp_epochs"] = 5
    b["training"]["seg_pos_weight"] = 60.0
    b["training"]["inter_pos_weight"] = 35.0
    b["training"]["use_dynamic_pos_weight"] = True
    b["training"]["use_dynamic_seg_pos_weight"] = True
    b["training"]["use_dynamic_inter_pos_weight"] = False
    b["training"]["min_pos_weight"] = 5.0
    b["training"]["max_pos_weight"] = 80.0
    profiles["profile_b_balanced"] = b

    # Profile C: 拓扑优先，冲 strict APLS/TopoIoU
    c = copy.deepcopy(base_cfg)
    c["training"]["learning_rate"] = 8e-5
    c["training"]["lambda_inter"] = 0.3
    c["training"]["lambda_orient"] = 0.3
    c["training"]["lambda_anchor"] = 0.15
    c["training"]["seg_warmup_epochs"] = 4
    c["training"]["multitask_ramp_epochs"] = 8
    c["training"]["seg_pos_weight"] = 50.0
    c["training"]["inter_pos_weight"] = 40.0
    c["training"]["use_dynamic_pos_weight"] = False
    c["training"]["use_dynamic_seg_pos_weight"] = False
    c["training"]["use_dynamic_inter_pos_weight"] = False
    c["evaluation"]["post_threshold"] = 0.25
    profiles["profile_c_topology"] = c

    return profiles


def main():
    parser = argparse.ArgumentParser(description="Run SCI2 sprint profiles.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true", help="Only generate profile configs")
    parser.add_argument("--run-train", action="store_true", help="Run training for each profile")
    parser.add_argument("--run-eval", action="store_true", help="Run eval/threshold/report for each profile")
    args = parser.parse_args()

    with open(os.path.join(args.workspace, args.config), "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.workspace, "experiments", f"sci2_sprint_{ts}")
    os.makedirs(out_root, exist_ok=True)

    profiles = build_profiles(base_cfg)
    manifest = []
    for name, cfg in profiles.items():
        run_dir = os.path.join(out_root, name)
        os.makedirs(run_dir, exist_ok=True)
        cfg["training"]["checkpoint_dir"] = os.path.join(run_dir, "checkpoints")
        cfg["training"]["log_dir"] = os.path.join(run_dir, "logs")
        cfg_path = os.path.join(run_dir, "config.yaml")
        write_yaml(cfg_path, cfg)

        item = {"profile": name, "config": cfg_path}
        if not args.prepare_only and args.run_train:
            run_cmd(f"\"{args.python}\" scripts/train.py --config \"{cfg_path}\"", cwd=args.workspace)
        if not args.prepare_only and args.run_eval:
            run_cmd(f"\"{args.python}\" scripts/eval_topology.py --config \"{cfg_path}\"", cwd=args.workspace)
            run_cmd(f"\"{args.python}\" scripts/search_post_threshold.py --config \"{cfg_path}\"", cwd=args.workspace)
            run_cmd(f"\"{args.python}\" scripts/plot_threshold_search.py", cwd=args.workspace)
            run_cmd(f"\"{args.python}\" scripts/build_paper_report.py", cwd=args.workspace)
        manifest.append(item)

    manifest_path = os.path.join(out_root, "sprint_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[Sprint] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

