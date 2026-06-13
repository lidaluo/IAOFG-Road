"""
从评估结果提取真实数值，生成论文表格（LaTeX）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "paper_materials" / "tables"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "N/A"


def extract_results() -> Dict[str, Dict[str, Any]]:
    return {
        "shanghai_opt": load_json(ROOT / "logs_shanghai_thick_optimized_final" / "eval" / "topology_eval.json"),
        "vegas_thin": load_json(ROOT / "logs_vegas_aoi2_eval" / "eval" / "vegas_topology_eval.json"),
        "vegas_thick": load_json(ROOT / "logs_vegas_aoi2_eval_thick" / "eval" / "vegas_topology_eval.json"),
    }


def create_result_table(res: Dict[str, Dict[str, Any]]) -> str:
    def row(name: str, d: Dict[str, Any]) -> str:
        valid = f"{d.get('strict_apls_valid_samples', 0)}/{d.get('num_samples', 0)}"
        return (
            f"{name} & {fmt(d.get('pixel_iou'))} & {fmt(d.get('pixel_f1'))} & "
            f"{fmt(d.get('topology_apls'))} & {fmt(d.get('topology_apls_strict'))} & {valid} \\\\"
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-city quantitative results of IAOF model.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Setting & Pixel IoU & Pixel F1 & APLS & Strict APLS & Valid Samples \\",
        r"\midrule",
        row("Shanghai (optimized final)", res["shanghai_opt"]),
        row("Vegas (original masks)", res["vegas_thin"]),
        row("Vegas (thick masks)", res["vegas_thick"]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def create_ablation_table(res: Dict[str, Dict[str, Any]]) -> str:
    thin = res["vegas_thin"]
    thick = res["vegas_thick"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation on data-format consistency (Vegas thin vs thick masks).}",
        r"\label{tab:ablation_data_consistency}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Setting & Pixel IoU & Pixel F1 & APLS & Strict APLS \\",
        r"\midrule",
        f"Vegas Thin & {fmt(thin.get('pixel_iou'))} & {fmt(thin.get('pixel_f1'))} & {fmt(thin.get('topology_apls'))} & {fmt(thin.get('topology_apls_strict'))} \\\\",
        f"Vegas Thick & {fmt(thick.get('pixel_iou'))} & {fmt(thick.get('pixel_f1'))} & {fmt(thick.get('topology_apls'))} & {fmt(thick.get('topology_apls_strict'))} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    res = extract_results()
    (TABLE_DIR / "result_tables.tex").write_text(create_result_table(res), encoding="utf-8")
    (TABLE_DIR / "ablation_tables.tex").write_text(create_ablation_table(res), encoding="utf-8")
    print(f"Saved: {(TABLE_DIR / 'result_tables.tex')}")
    print(f"Saved: {(TABLE_DIR / 'ablation_tables.tex')}")


if __name__ == "__main__":
    main()
