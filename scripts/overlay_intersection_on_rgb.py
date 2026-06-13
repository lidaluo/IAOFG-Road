import argparse
import os

import cv2
import numpy as np


def load_rgb(aoi_dir, sample_id):
    img_num = sample_id.replace("img", "")
    rgb_path = os.path.join(aoi_dir, "PS-RGB", f"SN3_roads_train_AOI_3_Paris_PS-RGB_img{img_num}.tif")
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"RGB tif not found: {rgb_path}")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="Overlay intersection heatmap onto RGB image.")
    parser.add_argument("--aoi-dir", type=str, required=True, help="SpaceNet AOI directory containing PS-RGB/")
    parser.add_argument("--labels-dir", type=str, default="data/spacenet_labels", help="Generated label directory")
    parser.add_argument("--sample-id", type=str, default=None, help="Single sample id (e.g., img10)")
    parser.add_argument("--num-samples", type=int, default=6, help="When sample-id is not given, export first N samples")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha")
    parser.add_argument("--out-dir", type=str, default="logs/intersection_overlay", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.sample_id:
        sample_ids = [args.sample_id]
    else:
        sample_ids = sorted(
            [d for d in os.listdir(args.labels_dir) if os.path.isdir(os.path.join(args.labels_dir, d))]
        )[: args.num_samples]

    for sid in sample_ids:
        inter_path = os.path.join(args.labels_dir, sid, "intersection.png")
        inter = cv2.imread(inter_path, cv2.IMREAD_GRAYSCALE)
        if inter is None:
            print(f"[Skip] Missing intersection map: {inter_path}")
            continue
        rgb = load_rgb(args.aoi_dir, sid)

        if rgb.shape[:2] != inter.shape[:2]:
            inter = cv2.resize(inter, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        inter_norm = inter.astype(np.float32) / 255.0
        heat_color = cv2.applyColorMap((inter_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        overlay = rgb.astype(np.float32).copy()
        mask = inter_norm > 0.05
        overlay[mask] = (1.0 - args.alpha) * overlay[mask] + args.alpha * heat_color.astype(np.float32)[mask]
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # 交叉口峰值点（可视化）
        local_max = cv2.dilate(inter, np.ones((9, 9), np.uint8))
        peaks = (inter >= local_max) & (inter > 40)
        ys, xs = np.where(peaks)
        for x, y in zip(xs, ys):
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 255), 1)

        out_path = os.path.join(args.out_dir, f"{sid}_intersection_overlay.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[OK] {sid} -> {out_path}")


if __name__ == "__main__":
    main()

