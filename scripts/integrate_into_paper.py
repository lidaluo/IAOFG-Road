"""
将方法、实验、图表清单整合为一个 markdown 草稿。
"""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures_dir", type=str, default="paper_materials/figures")
    ap.add_argument("--tables_dir", type=str, default="paper_materials/tables")
    ap.add_argument("--output", type=str, default="paper_draft/full_paper.md")
    args = ap.parse_args()

    figures_dir = (ROOT / args.figures_dir).resolve()
    tables_dir = (ROOT / args.tables_dir).resolve()
    out = (ROOT / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    method_md = (ROOT / "paper_materials" / "methodology_enhanced.md").read_text(encoding="utf-8")
    exp_md = (ROOT / "paper_materials" / "experiments_complete.md").read_text(encoding="utf-8")

    figure_list = sorted([p.name for p in figures_dir.glob("*.png")])
    table_list = sorted([p.name for p in tables_dir.glob("*.tex")])

    merged = []
    merged.append("# Full Paper Draft (Method + Experiments)\n")
    merged.append(method_md)
    merged.append("\n\n")
    merged.append(exp_md)
    merged.append("\n\n## 附录：已生成图表文件\n")
    for n in figure_list:
        merged.append(f"- `{figures_dir / n}`\n")
    merged.append("\n## 附录：已生成表格文件\n")
    for n in table_list:
        merged.append(f"- `{tables_dir / n}`\n")

    out.write_text("".join(merged), encoding="utf-8")
    print(f"Saved integrated draft: {out}")


if __name__ == "__main__":
    main()
