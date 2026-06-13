"""对比两份 topology_eval 风格 JSON（无需 pandas）。"""
import argparse
import json
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", help="基线 topology_eval.json")
    p.add_argument("optimized", help="优化后 topology_eval.json")
    args = p.parse_args()
    b = load(args.baseline)
    o = load(args.optimized)

    def ratio(d):
        n = d.get("num_samples") or 0
        vs = d.get("strict_apls_valid_samples") or 0
        return (vs / n * 100) if n else 0.0

    rb, ro = ratio(b), ratio(o)

    rows = [
        ("Strict APLS", b.get("topology_apls_strict", 0), o.get("topology_apls_strict", 0)),
        ("Mean APLS", b.get("topology_apls", 0), o.get("topology_apls", 0)),
        ("Pixel IoU", b.get("pixel_iou", 0), o.get("pixel_iou", 0)),
    ]

    print("优化前后对比")
    print("=" * 56)
    print(f"{'指标':<14} {'基线':>12} {'优化后':>12} {'变化':>12}")
    print("-" * 56)
    for name, vb, vo in rows:
        delta = vo - vb
        print(f"{name:<14} {vb:12.4f} {vo:12.4f} {delta:+12.4f}")
    print("-" * 56)
    print(
        f"{'有效样本比例':<14} {rb:11.1f}% {ro:11.1f}% {(ro - rb):+11.1f}%"
    )
    print("=" * 56)


if __name__ == "__main__":
    try:
        main()
    except OSError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
