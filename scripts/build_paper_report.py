import csv
import json
import os

import argparse


def safe_read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Build paper report markdown from eval outputs.")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=os.path.join("logs", "eval"),
        help="Directory containing topology_eval.json and threshold_search.csv.",
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    os.makedirs(eval_dir, exist_ok=True)
    topo = safe_read_json(os.path.join(eval_dir, "topology_eval.json")) or {}
    thresh = safe_read_csv(os.path.join(eval_dir, "threshold_search.csv"))

    best_thr = None
    if thresh:
        valid = [r for r in thresh if r.get("strict_apls") not in ("", "nan", "NaN", None)]
        if valid:
            best_thr = max(valid, key=lambda r: float(r["strict_apls"]))

    md = []
    md.append("# IAOF 论文结果草案")
    md.append("")
    md.append("## 主结果（当前）")
    md.append("")
    md.append("| 指标 | 数值 |")
    md.append("|---|---:|")
    for k in [
        "pixel_iou",
        "pixel_f1",
        "topology_apls_strict",
        "topology_apls",
        "topology_topoiou",
        "intersection_f1",
        "strict_apls_valid_samples",
    ]:
        if k in topo:
            md.append(f"| {k} | {topo[k]} |")

    md.append("")
    md.append("## 阈值敏感性（自动挑选）")
    md.append("")
    if best_thr is not None:
        md.append(
            f"- best threshold by strict_apls: `{best_thr['threshold']}` "
            f"(strict_apls={best_thr['strict_apls']}, valid={best_thr['strict_apls_valid_samples']})"
        )
    else:
        md.append("- 尚未找到有效 threshold_search 结果。")

    md.append("")
    md.append("## 图注建议（英文）")
    md.append("")
    md.append(
        "We evaluate IAOF-Topo on SpaceNet with both pixel-level and topology-level metrics. "
        "Threshold sensitivity demonstrates a robustness-accuracy trade-off; the selected threshold "
        "is chosen by strict APLS with sufficient valid-sample coverage."
    )

    out_path = os.path.join(eval_dir, "paper_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[Report] Saved: {out_path}")


if __name__ == "__main__":
    main()

