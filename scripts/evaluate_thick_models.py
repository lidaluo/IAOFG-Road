"""
厚数据集最佳模型评估（best epoch 19 vs final epoch 20）

仓库现有 `scripts/eval_topology.py` 不支持直接指定 checkpoint，
它会优先读取 `cfg['training']['checkpoint_dir']/model_best_val_iou.pth`。

因此本脚本做法：
1) 先评估 best：直接用现有 model_best_val_iou.pth
2) 再评估 final：临时用 model_epoch_20.pth 覆盖 model_best_val_iou.pth
3) 评估结束后恢复原 best 权重

然后把两次评估输出拷到：
  eval_results/thick_dataset/
并生成 summary.md。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_eval_outputs(src_eval_dir: Path, dst_out_dir: Path, prefix: str) -> None:
    dst_out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["topology_eval.json", "topology_eval.csv", "topology_eval_per_sample.csv"]:
        src = src_eval_dir / name
        if not src.is_file():
            continue
        dst = dst_out_dir / f"{prefix}_{name}"
        shutil.copy2(src, dst)


def _format_num(v) -> str:
    if v is None:
        return "N/A"
    try:
        fv = float(v)
        if fv != fv:  # NaN
            return "nan"
        return f"{fv:.4f}"
    except Exception:
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_shanghai_thick_eval.yaml")
    ap.add_argument("--output_dir", type=str, default="eval_results/thick_dataset")
    ap.add_argument("--best_prefix", type=str, default="eval_best_epoch19")
    ap.add_argument("--final_prefix", type=str, default="eval_final_epoch20")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    log_dir = Path(cfg["training"]["log_dir"])

    # 注意：eval_topology 固定从 checkpoint_dir/model_best_val_iou.pth 读
    best_model = checkpoint_dir / "model_best_val_iou.pth"
    final_model = checkpoint_dir / "model_epoch_20.pth"
    if not best_model.is_file():
        raise FileNotFoundError(f"best_model 不存在: {best_model}")
    if not final_model.is_file():
        raise FileNotFoundError(f"final_model 不存在: {final_model}")

    eval_dir = log_dir / "eval"
    best_json_out = out_dir / f"{args.best_prefix}_topology_eval.json"
    final_json_out = out_dir / f"{args.final_prefix}_topology_eval.json"

    # 1) best eval
    print("==> [1/2] Eval BEST (epoch 19) ...")
    _run(["python", str(repo_root / "scripts" / "eval_topology.py"), "--config", str(cfg_path)])
    _copy_eval_outputs(eval_dir, out_dir, args.best_prefix)
    if best_json_out.is_file():
        best = _load_json(best_json_out)
    else:
        best = {}

    # 2) final eval: temporarily replace model_best_val_iou.pth
    print("==> [2/2] Eval FINAL (epoch 20) ...")
    backup = checkpoint_dir / "_model_best_val_iou_backup_eval.pth"
    if backup.is_file():
        backup.unlink()
    shutil.copy2(best_model, backup)
    try:
        shutil.copy2(final_model, best_model)
        _run(["python", str(repo_root / "scripts" / "eval_topology.py"), "--config", str(cfg_path)])
        _copy_eval_outputs(eval_dir, out_dir, args.final_prefix)
    finally:
        shutil.copy2(backup, best_model)
        if backup.is_file():
            backup.unlink()

    final = _load_json(final_json_out) if final_json_out.is_file() else {}

    # summary.md
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Thick dataset evaluation summary\n\n")
        f.write(f"- config: `{cfg_path}`\n")
        f.write(f"- output: `{out_dir}`\n\n")
        f.write("| Model | topology_apls_strict | topology_apls | topology_topoiou | pixel_iou | pixel_f1 | intersection_f1 | strict_valid |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| best_epoch19 | {_format_num(best.get('topology_apls_strict'))} | "
            f"{_format_num(best.get('topology_apls'))} | {_format_num(best.get('topology_topoiou'))} | "
            f"{_format_num(best.get('pixel_iou'))} | {_format_num(best.get('pixel_f1'))} | "
            f"{_format_num(best.get('intersection_f1'))} | {_format_num(best.get('strict_apls_valid_samples'))} |\n"
        )
        f.write(
            f"| final_epoch20 | {_format_num(final.get('topology_apls_strict'))} | "
            f"{_format_num(final.get('topology_apls'))} | {_format_num(final.get('topology_topoiou'))} | "
            f"{_format_num(final.get('pixel_iou'))} | {_format_num(final.get('pixel_f1'))} | "
            f"{_format_num(final.get('intersection_f1'))} | {_format_num(final.get('strict_apls_valid_samples'))} |\n"
        )

    print(f"[OK] Eval outputs copied to: {out_dir}")
    print(f"[OK] Summary: {summary_path}")


if __name__ == "__main__":
    main()

