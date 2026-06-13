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


def main():
    parser = argparse.ArgumentParser(description="Run cross-city generalization evaluation.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train-aoi", type=str, required=True, help="Training AOI dir")
    parser.add_argument("--test-aois", type=str, nargs="+", required=True, help="One or more testing AOI dirs")
    parser.add_argument("--run-train", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only generate configs/manifest, do not execute")
    args = parser.parse_args()

    with open(os.path.join(args.workspace, args.config), "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.workspace, "experiments", f"cross_city_{ts}")
    os.makedirs(out_root, exist_ok=True)

    # Train config
    train_cfg = copy.deepcopy(base_cfg)
    train_cfg["data"]["aoi_dir"] = args.train_aoi
    train_cfg["training"]["checkpoint_dir"] = os.path.join(out_root, "train_checkpoints")
    train_cfg["training"]["log_dir"] = os.path.join(out_root, "train_logs")
    train_cfg_path = os.path.join(out_root, "train_config.yaml")
    write_yaml(train_cfg_path, train_cfg)

    if args.prepare_only:
        pass
    elif args.run_train:
        run_cmd(f"\"{args.python}\" scripts/train.py --config \"{train_cfg_path}\"", cwd=args.workspace)
    else:
        raise ValueError("Either set --run-train or --prepare-only.")

    results = []
    for test_aoi in args.test_aois:
        cfg = copy.deepcopy(train_cfg)
        cfg["data"]["aoi_dir"] = test_aoi
        city_name = os.path.basename(test_aoi.rstrip("\\/"))
        cfg["training"]["log_dir"] = os.path.join(out_root, f"eval_{city_name}")
        cfg_path = os.path.join(out_root, f"eval_config_{city_name}.yaml")
        write_yaml(cfg_path, cfg)
        if not args.prepare_only:
            run_cmd(f"\"{args.python}\" scripts/eval_topology.py --config \"{cfg_path}\"", cwd=args.workspace)
        results.append({"city": city_name, "config": cfg_path})

    with open(os.path.join(out_root, "cross_city_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[CrossCity] manifest saved: {os.path.join(out_root, 'cross_city_manifest.json')}")


if __name__ == "__main__":
    main()

