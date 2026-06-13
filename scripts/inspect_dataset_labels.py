import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_sample(aoi_dir, labels_dir, sample_id):
    img_num = sample_id.replace("img", "")
    rgb_path = os.path.join(aoi_dir, "PS-RGB", f"SN3_roads_train_AOI_3_Paris_PS-RGB_img{img_num}.tif")
    sample_dir = os.path.join(labels_dir, sample_id)
    mask_path = os.path.join(sample_dir, "mask.png")
    inter_path = os.path.join(sample_dir, "intersection.png")
    orient_path = os.path.join(sample_dir, "orientation.png")

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    inter = cv2.imread(inter_path, cv2.IMREAD_GRAYSCALE)
    orient = cv2.imread(orient_path, cv2.IMREAD_COLOR)
    if rgb is None or mask is None or inter is None or orient is None:
        raise FileNotFoundError(f"Sample files missing for {sample_id}")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    mask_f = mask.astype(np.float32) / 255.0
    inter_f = inter.astype(np.float32) / 255.0

    orient = orient.astype(np.float32)
    dx = orient[:, :, 0] / 127.5 - 1.0
    dy = orient[:, :, 1] / 127.5 - 1.0
    conf = orient[:, :, 2] / 255.0
    return rgb, mask_f, inter_f, dx, dy, conf, mask


def visualize_sample(out_path, sample_id, rgb, mask, inter, dx, dy, conf):
    road = mask > 0.5
    angle = np.arctan2(dy, dx)
    angle_vis = (angle + np.pi) / (2.0 * np.pi)
    angle_vis[~road] = 0.0
    mag = np.sqrt(dx * dx + dy * dy)
    mag[~road] = 0.0

    step = max(1, rgb.shape[0] // 32)
    yy, xx = np.mgrid[0 : rgb.shape[0] : step, 0 : rgb.shape[1] : step]
    qdx = dx[yy, xx]
    qdy = dy[yy, xx]
    qmask = road[yy, xx]
    qdx[~qmask] = 0
    qdy[~qmask] = 0

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{sample_id} RGB")
    axes[0, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Road Mask")
    axes[0, 2].imshow(inter, cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title("Intersection Heatmap")

    axes[1, 0].imshow(angle_vis, cmap="hsv", vmin=0, vmax=1)
    axes[1, 0].set_title("Orientation Angle (road only)")
    axes[1, 1].imshow(mag, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("Orientation Magnitude (road only)")
    axes[1, 2].imshow(rgb)
    axes[1, 2].quiver(xx, yy, qdx, -qdy, color="cyan", angles="xy", scale_units="xy", scale=1.2, width=0.002)
    axes[1, 2].set_title("Orientation Vector Overlay")

    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def sample_stats(mask, inter, dx, dy, conf, mask_u8):
    road = mask > 0.5
    road_ratio = float(np.mean(road))
    unique_vals = [int(v) for v in np.unique(mask_u8)]

    if np.any(road):
        norm = np.sqrt(dx[road] ** 2 + dy[road] ** 2)
        orient_norm_mean = float(np.mean(norm))
        orient_norm_std = float(np.std(norm))
        conf_road_mean = float(np.mean(conf[road]))
    else:
        orient_norm_mean = 0.0
        orient_norm_std = 0.0
        conf_road_mean = 0.0
    conf_bg_mean = float(np.mean(conf[~road])) if np.any(~road) else 0.0

    return {
        "road_ratio": road_ratio,
        "suggested_seg_pos_weight": float((1.0 - road_ratio) / max(road_ratio, 1e-6)),
        "mask_unique_values_u8": unique_vals,
        "intersection_max": float(np.max(inter)),
        "intersection_mean": float(np.mean(inter)),
        "orientation_norm_mean_on_road": orient_norm_mean,
        "orientation_norm_std_on_road": orient_norm_std,
        "confidence_mean_on_road": conf_road_mean,
        "confidence_mean_on_background": conf_bg_mean,
    }


def extract_intersection_peaks(inter, road_mask, nms_kernel=9, thr=0.3):
    hm = inter.astype(np.float32)
    local_max = cv2.dilate(hm, np.ones((nms_kernel, nms_kernel), np.uint8))
    peaks = (hm >= local_max - 1e-6) & (hm >= thr) & road_mask
    ys, xs = np.where(peaks)
    return np.stack([ys, xs], axis=1) if len(ys) > 0 else np.zeros((0, 2), dtype=np.int32)


def direction_consistency_stats(mask, inter, dx, dy, conf):
    road = mask > 0.5
    peaks = extract_intersection_peaks(inter, road, nms_kernel=9, thr=0.3)
    result = {
        "intersection_peak_count": int(len(peaks)),
        "dir_cos_to_nearest_intersection_mean": np.nan,
        "dir_cos_to_nearest_intersection_median": np.nan,
        "dir_norm_close_to_1_ratio": np.nan,
        "conf_road_is_one_ratio": np.nan,
        "conf_bg_is_zero_ratio": np.nan,
    }
    if not np.any(road):
        return result

    norm = np.sqrt(dx * dx + dy * dy)
    result["dir_norm_close_to_1_ratio"] = float(np.mean(np.abs(norm[road] - 1.0) < 0.1))
    result["conf_road_is_one_ratio"] = float(np.mean(conf[road] > 0.99))
    result["conf_bg_is_zero_ratio"] = float(np.mean(conf[~road] < 0.01)) if np.any(~road) else 1.0

    if len(peaks) == 0:
        return result

    ys, xs = np.where(road)
    vec_label = np.stack([dx[ys, xs], dy[ys, xs]], axis=1)
    pix = np.stack([ys, xs], axis=1)[:, None, :]  # [N,1,2]
    pk = peaks[None, :, :]  # [1,K,2]
    d2 = np.sum((pix - pk) ** 2, axis=2)  # [N,K]
    nn_idx = np.argmin(d2, axis=1)
    nearest = peaks[nn_idx]
    vec_to_inter = np.stack([nearest[:, 1] - xs, nearest[:, 0] - ys], axis=1).astype(np.float32)
    denom = np.linalg.norm(vec_to_inter, axis=1, keepdims=True) + 1e-6
    vec_to_inter = vec_to_inter / denom

    label_norm = np.linalg.norm(vec_label, axis=1, keepdims=True) + 1e-6
    vec_label = vec_label / label_norm
    cos = np.sum(vec_label * vec_to_inter, axis=1)
    result["dir_cos_to_nearest_intersection_mean"] = float(np.mean(cos))
    result["dir_cos_to_nearest_intersection_median"] = float(np.median(cos))
    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect SpaceNet labels with visualization and stats.")
    parser.add_argument("--aoi-dir", type=str, required=True)
    parser.add_argument("--labels-dir", type=str, default="data/spacenet_labels")
    parser.add_argument("--out-dir", type=str, default="logs/data_check")
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sample_ids = sorted([d for d in os.listdir(args.labels_dir) if os.path.isdir(os.path.join(args.labels_dir, d))])[: args.num_samples]
    all_stats = []
    for sid in sample_ids:
        rgb, mask, inter, dx, dy, conf, mask_u8 = load_sample(args.aoi_dir, args.labels_dir, sid)
        fig_path = os.path.join(args.out_dir, f"{sid}_inspection.png")
        visualize_sample(fig_path, sid, rgb, mask, inter, dx, dy, conf)
        st = sample_stats(mask, inter, dx, dy, conf, mask_u8)
        st.update(direction_consistency_stats(mask, inter, dx, dy, conf))
        st["sample_id"] = sid
        st["figure_path"] = fig_path
        all_stats.append(st)

    road_ratios = [x["road_ratio"] for x in all_stats]
    suggest_weights = [x["suggested_seg_pos_weight"] for x in all_stats]
    summary = {
        "num_samples_checked": len(all_stats),
        "avg_road_ratio": float(np.mean(road_ratios)) if road_ratios else 0.0,
        "avg_suggested_seg_pos_weight": float(np.mean(suggest_weights)) if suggest_weights else 0.0,
        "samples": all_stats,
    }
    out_json = os.path.join(args.out_dir, "label_inspection_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Inspect] saved summary: {out_json}")
    for s in all_stats:
        print(
            f"[Inspect] {s['sample_id']}: road_ratio={s['road_ratio']:.4f}, "
            f"pos_weight≈{s['suggested_seg_pos_weight']:.1f}, "
            f"norm_mean={s['orientation_norm_mean_on_road']:.3f}, "
            f"conf(road/bg)=({s['confidence_mean_on_road']:.3f}/{s['confidence_mean_on_background']:.3f}), "
            f"cos_to_inter(mean)={s['dir_cos_to_nearest_intersection_mean']}"
        )


if __name__ == "__main__":
    main()

