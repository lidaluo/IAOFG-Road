"""
论文材料整理脚本
生成方法描述、实验结果、可视化样本与对比实验设计草稿。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SampleRow:
    sample_id: str
    apls: float
    strict_apls: Optional[float]
    pixel_iou: float


class PaperMaterialsOrganizer:
    def __init__(self, eval_results_dir: str = "eval_results/final_optimized") -> None:
        self.eval_results_dir = Path(eval_results_dir)
        self.paper_dir = Path("paper_materials")
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        self.results = self._load_evaluation_results()
        self.per_sample_rows = self._load_per_sample_rows()

    def _candidate_result_jsons(self) -> List[Path]:
        return [
            self.eval_results_dir / "final_results.json",
            self.eval_results_dir / "topology_eval.json",
            Path("logs_shanghai_thick_optimized_final/eval/topology_eval.json"),
        ]

    def _load_evaluation_results(self) -> Dict:
        for p in self._candidate_result_jsons():
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
        return {}

    def _candidate_per_sample_csvs(self) -> List[Path]:
        return [
            self.eval_results_dir / "topology_eval_per_sample.csv",
            Path("logs_shanghai_thick_optimized_final/eval/topology_eval_per_sample.csv"),
        ]

    def _to_float(self, v: str) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _to_optional_float(self, v: str) -> Optional[float]:
        if v is None:
            return None
        vv = str(v).strip().lower()
        if vv in {"", "nan", "none"}:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _load_per_sample_rows(self) -> List[SampleRow]:
        csv_path = None
        for p in self._candidate_per_sample_csvs():
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            return []

        rows: List[SampleRow] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    SampleRow(
                        sample_id=row.get("sample_id", ""),
                        apls=self._to_float(row.get("apls", 0.0)),
                        strict_apls=self._to_optional_float(row.get("strict_apls", "")),
                        pixel_iou=self._to_float(row.get("pixel_iou", 0.0)),
                    )
                )
        return rows

    def generate_method_description(self) -> str:
        method_desc = """# Methodology (Draft)

## 3.1 Overview
我们提出一种**交叉口驱动（intersection-driven）**的道路拓扑提取方法。模型先预测分割、交叉口热力图与方向场，再通过后处理重建道路图（节点与边），从而兼顾像素级覆盖与拓扑连通性。

## 3.2 Network and Multi-task Learning
- 主干网络：`swin_tiny`
- 任务头：道路分割、交叉口检测、方向场回归
- 损失权重（Shanghai thick 配置）：`lambda_seg=1.0`, `lambda_inter=0.05`, `lambda_orient=0.05`, `lambda_anchor=0.01`

## 3.3 Topology Reconstruction
后处理按照以下流程执行：
1. 对分割概率图阈值化（`post_threshold`）
2. 对交叉口热图进行 NMS（`post_nms_size`）
3. 在方向场引导下进行节点连接（`endpoint_dist`, `dir_stop_eps`, `angle_step`, `step_size`）
4. 删除过短路径（`min_path_len`）并输出图结构

## 3.4 Optimized Post-processing Parameters
- `post_threshold=0.26`
- `post_nms_size=3`
- `min_intersections=2`
- `endpoint_dist=10.0`
- `dir_stop_eps=0.2`
- `min_path_len=12`
- `angle_step=45`
- `step_size=1.0`

## 3.5 Training/Evaluation Setup
- 数据：SpaceNet4 Shanghai（thick mask 版本）
- 评估：APLS / Strict APLS / Pixel IoU / Pixel F1 / TopoIoU
- 推理权重：`checkpoints_shanghai_thick/model_best_val_iou.pth`（Epoch 19）
"""
        method_file = self.paper_dir / "methodology.md"
        method_file.write_text(method_desc, encoding="utf-8")
        print(f"[OK] 方法描述: {method_file}")
        return method_desc

    def _safe_ratio(self, num: float, den: float) -> float:
        return (num / den * 100.0) if den else 0.0

    def generate_experiment_results(self) -> Optional[str]:
        if not self.results:
            print("[WARN] 未找到评估结果 JSON，跳过实验结果文稿。")
            return None

        topology_apls = float(self.results.get("topology_apls", 0.0))
        strict_apls = float(self.results.get("topology_apls_strict", 0.0))
        pixel_iou = float(self.results.get("pixel_iou", 0.0))
        pixel_f1 = float(self.results.get("pixel_f1", 0.0))
        topo_iou = float(self.results.get("topology_topoiou", 0.0))
        valid = int(self.results.get("strict_apls_valid_samples", 0))
        total = int(self.results.get("num_samples", 0))
        valid_ratio = self._safe_ratio(valid, total)

        results_md = f"""# Experimental Results (Draft)

## 4.1 Dataset and Metrics
- Dataset: SpaceNet4 Shanghai (thick mask)
- Metrics: Topology APLS, Strict APLS, Pixel IoU, Pixel F1, TopoIoU

## 4.2 Final Quantitative Results
| Metric | Value |
|--------|-------|
| Topology APLS | {topology_apls:.4f} |
| Strict APLS | {strict_apls:.4f} |
| Pixel IoU | {pixel_iou:.4f} |
| Pixel F1 | {pixel_f1:.4f} |
| Topology TopoIoU | {topo_iou:.4f} |
| Valid Samples (strict) | {valid}/{total} |
| Valid Ratio | {valid_ratio:.1f}% |

## 4.3 Notes for Paper Writing
1. 报告主指标时建议同时给出 APLS 与 Strict APLS。
2. 对 Strict APLS 需解释“可计算样本比例（Valid Ratio）”。
3. 讨论部分建议给出典型成功/失败样例图，解释拓扑断裂与误连接来源。
"""

        results_file = self.paper_dir / "experimental_results.md"
        results_file.write_text(results_md, encoding="utf-8")
        print(f"[OK] 实验结果文稿: {results_file}")
        return results_md

    def generate_topology_visualization_manifest(self, num_samples: int = 6) -> List[Dict]:
        if not self.per_sample_rows:
            print("[WARN] 未找到 per-sample CSV，无法生成可视化样本清单。")
            return []

        rows = list(self.per_sample_rows)
        rows_sorted_apls = sorted(rows, key=lambda x: x.apls, reverse=True)
        top = rows_sorted_apls[:2]
        low = sorted(rows, key=lambda x: x.apls)[:2]
        valid_strict = [r for r in rows if r.strict_apls is not None][:2]

        selected: List[SampleRow] = []
        seen = set()
        for group in (top, low, valid_strict, rows[: max(0, num_samples)]):
            for r in group:
                if r.sample_id and r.sample_id not in seen:
                    selected.append(r)
                    seen.add(r.sample_id)
                if len(selected) >= num_samples:
                    break
            if len(selected) >= num_samples:
                break

        payload = [
            {
                "sample_id": r.sample_id,
                "apls": r.apls,
                "strict_apls": r.strict_apls,
                "pixel_iou": r.pixel_iou,
            }
            for r in selected
        ]
        out = self.paper_dir / "visualization_samples.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] 可视化样本清单: {out} (n={len(payload)})")
        return payload

    def design_comparison_experiments(self) -> str:
        comparison_design = """# Comparison Experiment Design (RoadTracer vs Sat2Graph)

## 5.1 Compared Methods
- RoadTracer (CVPR 2018): graph-growing
- Sat2Graph (ECCV 2020): graph-tensor prediction
- Ours: intersection-driven topology extraction

## 5.2 Fair Protocol
1. Same dataset/split (SpaceNet4 Shanghai, 与本工作一致)
2. Same evaluation script for APLS/Strict APLS
3. Same image resolution policy and preprocessing
4. Report runtime (sec/image) on same hardware

## 5.3 What to Compare
- Main: APLS, Strict APLS
- Secondary: Pixel IoU, valid strict sample ratio
- Efficiency: average inference latency
- Failure modes: broken roads / false links / missed intersections

## 5.4 Implementation Checklist
- [ ] 将 RoadTracer/Sat2Graph 输出统一转换为当前评估脚本可读图结构
- [ ] 在相同验证集生成 per-sample 结果
- [ ] 输出统一对比表与案例可视化
"""
        comparison_file = self.paper_dir / "comparison_experiment_design.md"
        comparison_file.write_text(comparison_design, encoding="utf-8")
        print(f"[OK] 对比实验设计: {comparison_file}")
        return comparison_design

    def create_summary_report(self) -> str:
        valid = int(self.results.get("strict_apls_valid_samples", 0)) if self.results else 0
        total = int(self.results.get("num_samples", 0)) if self.results else 0
        ratio = self._safe_ratio(valid, total)
        summary = f"""# Paper Materials Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Files
- `paper_materials/methodology.md`
- `paper_materials/experimental_results.md`
- `paper_materials/visualization_samples.json`
- `paper_materials/comparison_experiment_design.md`

## Key Metrics
- Topology APLS: {float(self.results.get("topology_apls", 0.0)) if self.results else 0.0:.4f}
- Strict APLS: {float(self.results.get("topology_apls_strict", 0.0)) if self.results else 0.0:.4f}
- Valid strict samples: {valid}/{total} ({ratio:.1f}%)

## Next
1. 运行 `scripts/generate_topology_figures.py` 生成论文图
2. 在论文中填充方法描述与结果表
3. 按 `comparison_experiment_design.md` 补齐 baseline 实验
"""
        summary_file = self.paper_dir / "summary.md"
        summary_file.write_text(summary, encoding="utf-8")
        print(f"[OK] 总结报告: {summary_file}")
        return summary

    def run_all(self, num_samples: int = 6) -> bool:
        print("=" * 60)
        print("开始整理论文材料")
        print("=" * 60)
        self.generate_method_description()
        self.generate_experiment_results()
        self.generate_topology_visualization_manifest(num_samples=num_samples)
        self.design_comparison_experiments()
        self.create_summary_report()
        print("=" * 60)
        print(f"完成，输出目录: {self.paper_dir.resolve()}")
        print("=" * 60)
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="整理论文材料")
    parser.add_argument("--eval_dir", default="eval_results/final_optimized", help="评估结果目录")
    parser.add_argument("--num_samples", type=int, default=6, help="可视化样本数")
    args = parser.parse_args()
    PaperMaterialsOrganizer(eval_results_dir=args.eval_dir).run_all(num_samples=args.num_samples)


if __name__ == "__main__":
    main()
