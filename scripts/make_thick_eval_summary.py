from __future__ import annotations

import json
from pathlib import Path


def _fmt(v) -> str:
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
    out_dir = Path("eval_results/thick_dataset")
    best_p = out_dir / "eval_best_epoch19_topology_eval.json"
    final_p = out_dir / "eval_final_epoch20_topology_eval.json"
    if not best_p.is_file() or not final_p.is_file():
        raise FileNotFoundError(f"Missing eval json: {best_p} / {final_p}")

    best = json.load(open(best_p, "r", encoding="utf-8"))
    final = json.load(open(final_p, "r", encoding="utf-8"))

    summary = []
    summary.append("# Thick dataset evaluation summary")
    summary.append("")
    summary.append("| Model | topology_apls_strict | topology_apls | topology_topoiou | pixel_iou | pixel_f1 | intersection_f1 | strict_valid |")
    summary.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    summary.append(
        "| best_epoch19 | "
        + _fmt(best.get("topology_apls_strict"))
        + " | "
        + _fmt(best.get("topology_apls"))
        + " | "
        + _fmt(best.get("topology_topoiou"))
        + " | "
        + _fmt(best.get("pixel_iou"))
        + " | "
        + _fmt(best.get("pixel_f1"))
        + " | "
        + _fmt(best.get("intersection_f1"))
        + " | "
        + str(best.get("strict_apls_valid_samples"))
        + " |"
    )
    summary.append(
        "| final_epoch20 | "
        + _fmt(final.get("topology_apls_strict"))
        + " | "
        + _fmt(final.get("topology_apls"))
        + " | "
        + _fmt(final.get("topology_topoiou"))
        + " | "
        + _fmt(final.get("pixel_iou"))
        + " | "
        + _fmt(final.get("pixel_f1"))
        + " | "
        + _fmt(final.get("intersection_f1"))
        + " | "
        + str(final.get("strict_apls_valid_samples"))
        + " |"
    )

    out_md = out_dir / "summary.md"
    out_md.write_text("\n".join(summary), encoding="utf-8")
    print(f"[OK] wrote {out_md.resolve()}")


if __name__ == "__main__":
    main()

