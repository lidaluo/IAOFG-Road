"""从 topology_eval.json（或同结构 JSON）打印论文用摘要与 Markdown 表格。"""
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_path", help="topology_eval.json 或 final_results.json")
    args = p.parse_args()
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = data.get("num_samples") or 0
    vs = data.get("strict_apls_valid_samples") or 0
    ratio = (vs / n * 100) if n else 0.0

    print("最终评估结果")
    print("=" * 40)
    print(f"Strict APLS: {data.get('topology_apls_strict', 0):.4f}")
    print(f"Mean APLS: {data.get('topology_apls', 0):.4f}")
    print(f"Pixel IoU: {data.get('pixel_iou', 0):.4f}")
    print(f"Pixel F1: {data.get('pixel_f1', 0):.4f}")
    print(f"TopoIoU: {data.get('topology_topoiou', 0):.4f}")
    print(f"有效样本: {vs}/{n}")
    print(f"有效比例: {ratio:.1f}%")
    print("=" * 40)
    print()
    print("论文结果表格（Markdown）")
    print("| 指标 | 数值 | 说明 |")
    print("|------|------|------|")
    print(f"| Strict APLS | {data.get('topology_apls_strict', 0):.4f} | 严格拓扑相似度 |")
    print(f"| Mean APLS | {data.get('topology_apls', 0):.4f} | 平均拓扑相似度 |")
    print(f"| Pixel IoU | {data.get('pixel_iou', 0):.4f} | 像素交并比 |")
    print(f"| 有效样本比例 | {ratio:.1f}% | 可计算 strict APLS 的样本比例 |")


if __name__ == "__main__":
    try:
        main()
    except OSError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
