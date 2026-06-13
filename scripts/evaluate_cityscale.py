"""
City-Scale 测试集评估：IAOF 全局推理 +（可选）Local Query 精修；
输出 TOPO-P、TOPO-R、TOPO-F1（栅格缓冲带口径）与 APLS。
IAOF++：可选整图方向 bin 累加 + iterative_endpoint_completion（见 configs/cityscale_iaofpp.yaml）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.cityscale_dataset import load_cityscale_split, resolve_cityscale_rgb_path
from metrics.metrics import MetricsCalculator
from models.local_query_decoder import LocalQueryDecoder
from models.road_extraction_model import RoadExtractionModel
from postprocess.centerline_from_mask import mask_to_centerline_graph
from postprocess.graph_refinement import refine_graph_with_local_decoder
from utils.cityscale_graph import (
    ensure_edge_lengths,
    load_cityscale_gt_graph,
    relabel_graph_continuous,
    save_nx_graph_geojson,
    save_nx_graph_pickle,
)
from utils import cv_io
from utils.orient_bins import logits_to_expected_dxdy


def _save_seg_orient_npy(
    export_dir: str,
    region_index: int,
    prob_hw: np.ndarray,
    orient_logits_khw: np.ndarray | None,
) -> None:
    """
    保存 iaof_postprocessing / demo 所需：
    - region_{t}_seg.npy  float32 H×W 分割概率
    - region_{t}_orient.npy  float32 H×W×K 方向 softmax（无方向头则跳过）
    """
    os.makedirs(export_dir, exist_ok=True)
    t = int(region_index)
    seg_path = os.path.join(export_dir, f"region_{t}_seg.npy")
    np.save(seg_path, prob_hw.astype(np.float32))
    if orient_logits_khw is None or orient_logits_khw.size == 0:
        return
    z = np.asarray(orient_logits_khw, dtype=np.float64)
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    p = e / np.sum(e, axis=0, keepdims=True)
    orient_hwk = np.transpose(p.astype(np.float32), (1, 2, 0))
    np.save(os.path.join(export_dir, f"region_{t}_orient.npy"), orient_hwk)


def _infer_full_tile(
    model: RoadExtractionModel,
    device: torch.device,
    tile_index: int,
    cityscale_root: str,
    patch_size: int,
    full_size: int,
    encoder_hw: tuple[int, int],
    processed_subdir: str,
    images_subdir: str,
    sat_subdir: str,
) -> np.ndarray:
    """4×4 块推理分割概率，拼接为 full_size 概率图。"""
    proc = os.path.join(cityscale_root, processed_subdir)
    grid = full_size // patch_size
    acc = np.zeros((full_size, full_size), dtype=np.float32)
    cnt = np.zeros_like(acc)
    rgb_path = resolve_cityscale_rgb_path(
        cityscale_root, tile_index, sat_subdir=sat_subdir, images_subdir=images_subdir
    )

    model.eval()
    with torch.no_grad():
        for gi in range(grid):
            for gj in range(grid):
                y0, x0 = gi * patch_size, gj * patch_size
                y1, x1 = y0 + patch_size, x0 + patch_size
                road = cv_io.imread_gray(os.path.join(proc, f"road_mask_{tile_index}.png"))
                if road is None:
                    raise FileNotFoundError(f"缺少 road_mask_{tile_index}.png")
                if rgb_path:
                    bgr = cv_io.imread_bgr(rgb_path)
                    if bgr is None:
                        raise FileNotFoundError(f"无法读取 RGB: {rgb_path}")
                    img = cv_io.bgr_to_rgb_u8(bgr).astype(np.float32) / 255.0
                else:
                    r = (road > 0).astype(np.float32)
                    img = np.stack([r, r, r], axis=-1)
                crop = img[y0:y1, x0:x1]
                eh, ew = int(encoder_hw[0]), int(encoder_hw[1])
                tin = cv_io.resize(crop, (ew, eh), linear=True)
                t = torch.from_numpy(np.transpose(tin, (2, 0, 1)))[None, ...].float().to(device)
                logits = model(t)["segmentation"]
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                prob_up = cv_io.resize(prob, (patch_size, patch_size), linear=True)
                acc[y0:y1, x0:x1] += prob_up
                cnt[y0:y1, x0:x1] += 1.0
    out = acc / np.maximum(cnt, 1e-6)
    return out


def _infer_full_tile_seg_orient(
    model: RoadExtractionModel,
    device: torch.device,
    tile_index: int,
    cityscale_root: str,
    patch_size: int,
    full_size: int,
    encoder_hw: tuple[int, int],
    processed_subdir: str,
    images_subdir: str,
    sat_subdir: str,
    orient_num_bins: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """返回 (seg_prob HxW, orient_logits_acc KxH|W 或 None)。"""
    proc = os.path.join(cityscale_root, processed_subdir)
    grid = full_size // patch_size
    acc = np.zeros((full_size, full_size), dtype=np.float32)
    cnt = np.zeros_like(acc)
    acc_o = None
    cnt_o = None
    if orient_num_bins > 0:
        acc_o = np.zeros((orient_num_bins, full_size, full_size), dtype=np.float32)
        cnt_o = np.zeros((full_size, full_size), dtype=np.float32)

    rgb_path = resolve_cityscale_rgb_path(
        cityscale_root, tile_index, sat_subdir=sat_subdir, images_subdir=images_subdir
    )

    model.eval()
    with torch.no_grad():
        for gi in range(grid):
            for gj in range(grid):
                y0, x0 = gi * patch_size, gj * patch_size
                y1, x1 = y0 + patch_size, x0 + patch_size
                road = cv_io.imread_gray(os.path.join(proc, f"road_mask_{tile_index}.png"))
                if road is None:
                    raise FileNotFoundError(f"缺少 road_mask_{tile_index}.png")
                if rgb_path:
                    bgr = cv_io.imread_bgr(rgb_path)
                    if bgr is None:
                        raise FileNotFoundError(f"无法读取 RGB: {rgb_path}")
                    img = cv_io.bgr_to_rgb_u8(bgr).astype(np.float32) / 255.0
                else:
                    r = (road > 0).astype(np.float32)
                    img = np.stack([r, r, r], axis=-1)
                crop = img[y0:y1, x0:x1]
                eh, ew = int(encoder_hw[0]), int(encoder_hw[1])
                tin = cv_io.resize(crop, (ew, eh), linear=True)
                t = torch.from_numpy(np.transpose(tin, (2, 0, 1)))[None, ...].float().to(device)
                out = model(t)
                prob = torch.sigmoid(out["segmentation"])[0, 0].cpu().numpy()
                prob_up = cv_io.resize(prob, (patch_size, patch_size), linear=True)
                acc[y0:y1, x0:x1] += prob_up
                cnt[y0:y1, x0:x1] += 1.0
                if acc_o is not None and out["orientation"].shape[1] == orient_num_bins:
                    ol = out["orientation"][0].float().cpu().numpy()
                    for k in range(orient_num_bins):
                        sl = cv_io.resize(ol[k], (patch_size, patch_size), linear=True)
                        acc_o[k, y0:y1, x0:x1] += sl
                    cnt_o[y0:y1, x0:x1] += 1.0

    prob_full = acc / np.maximum(cnt, 1e-6)
    if acc_o is not None and cnt_o is not None and float(cnt_o.max()) > 0:
        cs = np.maximum(cnt_o, 1e-6)
        acc_o = acc_o / cs[np.newaxis, :, :]
        return prob_full, acc_o
    return prob_full, None


def _region_indices_for_split(split_json: dict, split: str) -> list[int]:
    """data_split.json 中的区域索引列表（整 tile，非 patch_id）。"""
    s = split.strip().lower()
    if s == "test":
        raw = split_json.get("test", [])
    elif s in ("valid", "val", "validation"):
        raw = split_json.get("valid", split_json.get("val", []))
    elif s == "train":
        raw = split_json.get("train", [])
    else:
        raise ValueError(f"未知 split={split!r}，请用 train / valid / test")
    out: list[int] = []
    for x in raw:
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, str) and "_" in x:
            out.append(int(x.split("_", 1)[0]))
        elif isinstance(x, str) and x.isdigit():
            out.append(int(x))
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser(
        description="City-Scale 整图推理 + 骨架矢量化 + TOPO-P/R/F1 与 APLS（metrics.MetricsCalculator）。"
    )
    parser.add_argument("--config", type=str, default="configs/cityscale_iaofpp.yaml")
    parser.add_argument("--global-ckpt", type=str, default=None, help="RoadExtractionModel state_dict")
    parser.add_argument("--local-ckpt", type=str, default=None, help="LocalQueryDecoder state_dict（可选）")
    parser.add_argument("--no-refine", action="store_true", help="不做局部精修")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="评估划分：valid=验证集（与训练 val 同源区域列表），test=官方 test，train=训练区域",
    )
    parser.add_argument(
        "--save-graph-dir",
        type=str,
        default=None,
        help="若指定，则每个 region 保存 pred 路网：region_{t}_pred_graph.pkl + .geojson",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="若指定，写入 mean TOPO-P/R/F1、APLS 及逐 tile 分数的 JSON",
    )
    parser.add_argument(
        "--export-npy-dir",
        type=str,
        default=None,
        help="保存每区域 region_{t}_seg.npy（H×W 概率）与 region_{t}_orient.npy（H×W×K softmax，需 IAOF++ 整图方向推理）",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help="逗号分隔的 region 索引；指定时只处理这些 tile（覆盖按 --split 枚举的列表），便于增量导出 npy 或只评子集。",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="在最终区域列表上截断为前 N 个（与 --regions 或 split 联用）。",
    )
    parser.add_argument(
        "--iaofpp-pipeline",
        action="store_true",
        help="使用 IAOF++：整图方向 bin 推理 + 迭代端点补全（需配置 model.orient_num_bins 与权重一致）",
    )
    parser.add_argument(
        "--per-tile-jsonl",
        type=str,
        default=None,
        help="每评完一张 tile 追加一行 JSON（含 region、TOPO-P/R/F1、APLS、耗时秒），便于长任务中途查看。",
    )
    parser.add_argument(
        "--append-run-date",
        action="store_true",
        help="在 --summary-json 与 --per-tile-jsonl 的文件名（扩展名前）追加今日 ISO 日期，例如 ..._2026-05-05.json。",
    )
    args = parser.parse_args()

    if args.append_run_date:
        _tag = date.today().isoformat()

        def _with_date(path: str | None) -> str | None:
            if not path:
                return path
            root, ext = os.path.splitext(path)
            if root.endswith(_tag) or root.endswith("_" + _tag):
                return path
            return f"{root}_{_tag}{ext}"

        args.summary_json = _with_date(args.summary_json)
        args.per_tile_jsonl = _with_date(args.per_tile_jsonl)

    os.chdir(PROJECT_ROOT)
    # 经 PowerShell「2>&1 | Tee-Object」等管道重定向时 stdout 非 TTY，默认块缓冲会导致终端长时间无输出。
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    if not cv_io.cv2_io_ok():
        try:
            import cv2 as _cv_dbg

            _cv_path = getattr(_cv_dbg, "__file__", "?")
        except Exception:
            _cv_path = "?"
        print(
            f"[Eval][Warn] cv2 不完整（__file__={_cv_path}），读图/缩放与 TOPO 栅格化已改用 PIL + skimage；"
            f"数值与标准 OpenCV 可能略有差异。建议修复: pip install opencv-python-headless",
            flush=True,
        )
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = cfg["data"]["root"]
    split_path = cfg["data"].get("split_json") or os.path.join(root, "data_split.json")
    sp = load_cityscale_split(split_path)
    if args.regions:
        test_regions = sorted({int(x.strip()) for x in args.regions.split(",") if x.strip()})
        print(f"[Eval] 使用 --regions 显式列表，共 {len(test_regions)} 个 tile", flush=True)
    else:
        test_regions = _region_indices_for_split(sp, args.split)
    if args.max_tiles is not None:
        test_regions = test_regions[: int(args.max_tiles)]
    print(
        f"[Eval] split={args.split!r} 待处理区域数={len(test_regions)}  indices(前20)={test_regions[:20]}",
        flush=True,
    )
    patch_size = int(cfg["data"].get("patch_size", 512))
    full_size = int(cfg["data"].get("full_image_size", 2048))
    enc = tuple(cfg["data"].get("encoder_input_size", (224, 224)))

    _ckpt_dir = cfg.get("evaluation", {}).get("checkpoint_dir") or cfg["training"]["checkpoint_dir"]
    g_ckpt = args.global_ckpt or os.path.join(_ckpt_dir, "model_best_val_iou.pth")
    if not os.path.isfile(g_ckpt):
        g_ckpt = os.path.join(_ckpt_dir, "model_latest.pth")
    print(f"[Eval] 全局权重: {g_ckpt}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] device={device}", flush=True)
    swin_img = cfg["model"].get("swin_img_size")
    if swin_img is not None:
        swin_img = int(swin_img)
    orient_bins = int(cfg.get("model", {}).get("orient_num_bins", 0))
    model = RoadExtractionModel(
        encoder=cfg["model"]["encoder"],
        num_classes=1,
        input_size=enc,
        swin_img_size=swin_img,
        orient_num_bins=orient_bins,
    ).to(device)
    _sd = torch.load(g_ckpt, map_location=device)
    _incompatible = model.load_state_dict(_sd, strict=False)
    if _incompatible.missing_keys or _incompatible.unexpected_keys:
        print(
            f"[Eval] load_state_dict(strict=False): missing={len(_incompatible.missing_keys)} "
            f"unexpected={len(_incompatible.unexpected_keys)}",
            flush=True,
        )
    model.eval()

    lqd = None
    if not args.no_refine:
        lpath = args.local_ckpt or cfg.get("local_query", {}).get(
            "checkpoint_path", os.path.join(cfg["training"]["checkpoint_dir"], "local_query_decoder.pth")
        )
        if os.path.isfile(lpath):
            lqd = LocalQueryDecoder(in_channels=4, base=32).to(device)
            lqd.load_state_dict(torch.load(lpath, map_location=device))
            lqd.eval()
        else:
            print(f"[Warn] 未找到局部权重 {lpath}，跳过精修。", flush=True)

    thr = float(cfg.get("evaluation", {}).get("post_threshold", 0.25))
    buf = int(cfg.get("evaluation", {}).get("topo_f1_buffer_px", 5))
    calc = MetricsCalculator(max_gt_nodes_for_apls=int(cfg.get("evaluation", {}).get("max_gt_nodes_apls", 64)))
    sat_sub = cfg["data"].get("sat_images_subdir", "20cities")
    img_sub = cfg["data"].get("images_subdir", "images")

    apls_list = []
    topo_p_list: list[float] = []
    topo_r_list: list[float] = []
    topo_f1_list: list[float] = []
    per_tile_rows: list[dict] = []
    graph_dir = args.save_graph_dir
    if graph_dir and not os.path.isabs(graph_dir):
        graph_dir = os.path.join(PROJECT_ROOT, graph_dir)
    export_npy_dir = args.export_npy_dir
    if export_npy_dir and not os.path.isabs(export_npy_dir):
        export_npy_dir = os.path.join(PROJECT_ROOT, export_npy_dir)
    use_iaofpp_eval = bool(args.iaofpp_pipeline or cfg.get("iaofpp", {}).get("eval_use_iaofpp_pipeline", False))
    ipp = cfg.get("iaofpp", {})

    n_total = len(test_regions)
    per_tile_jsonl_path = args.per_tile_jsonl
    if per_tile_jsonl_path and not os.path.isabs(per_tile_jsonl_path):
        per_tile_jsonl_path = os.path.join(PROJECT_ROOT, per_tile_jsonl_path)
    per_tile_f = None
    if per_tile_jsonl_path:
        os.makedirs(os.path.dirname(per_tile_jsonl_path) or ".", exist_ok=True)
        per_tile_f = open(per_tile_jsonl_path, "w", encoding="utf-8")

    for k, t in enumerate(test_regions, start=1):
        t0 = time.perf_counter()
        if use_iaofpp_eval and orient_bins > 0:
            prob, orient_acc = _infer_full_tile_seg_orient(
                model,
                device,
                t,
                root,
                patch_size,
                full_size,
                enc,
                cfg["data"].get("processed_subdir", "processed"),
                img_sub,
                sat_sub,
                orient_bins,
            )
        else:
            prob = _infer_full_tile(
                model,
                device,
                t,
                root,
                patch_size,
                full_size,
                enc,
                cfg["data"].get("processed_subdir", "processed"),
                img_sub,
                sat_sub,
            )
            orient_acc = None

        if export_npy_dir:
            _save_seg_orient_npy(export_npy_dir, int(t), prob, orient_acc)
            if orient_acc is not None:
                print(
                    f"[Export] region_{int(t)}_seg.npy + region_{int(t)}_orient.npy -> {export_npy_dir}",
                    flush=True,
                )
            else:
                print(
                    f"[Export] region_{int(t)}_seg.npy（无 orient，请加 --iaofpp-pipeline 或 yaml iaofpp.eval_use_iaofpp_pipeline）",
                    flush=True,
                )

        pred_mask = (prob > thr).astype(np.uint8) * 255
        rgb_path = resolve_cityscale_rgb_path(root, t, sat_subdir=sat_sub, images_subdir=img_sub)
        rgb = None
        if rgb_path:
            bgr = cv_io.imread_bgr(rgb_path)
            if bgr is None:
                raise FileNotFoundError(f"无法读取 RGB: {rgb_path}")
            rgb = cv_io.bgr_to_rgb_u8(bgr).astype(np.float32) / 255.0

        if lqd is not None:
            pred_graph = refine_graph_with_local_decoder(pred_mask, rgb, lqd, device)
        else:
            pred_graph = mask_to_centerline_graph(pred_mask, open_kernel=0, merge_node_dist=4.0, use_medial_axis=True)

        if (
            ipp.get("use_iterative_completion", False)
            and orient_acc is not None
            and float(np.max(orient_acc)) != 0.0
        ):
            from postprocess.local_path_completer import iterative_endpoint_completion

            ot = torch.from_numpy(orient_acc.astype(np.float32)).unsqueeze(0).to(device)
            dxy = logits_to_expected_dxdy(ot)[0].cpu().numpy()
            dir_hw2 = np.transpose(dxy, (1, 2, 0)).astype(np.float32)
            pred_graph = iterative_endpoint_completion(
                pred_graph,
                (pred_mask > 0).astype(np.float32),
                dir_hw2,
                max_steps=int(ipp.get("completion_max_steps", 5)),
                step_size=float(ipp.get("completion_step_px", 2.0)),
                connect_dist=float(ipp.get("completion_connect_dist", 22.0)),
            )

        pred_graph = relabel_graph_continuous(pred_graph)
        pred_graph = ensure_edge_lengths(pred_graph)

        gt_path = os.path.join(root, "20cities", f"region_{t}_refine_gt_graph.p")
        if not os.path.isfile(gt_path):
            print(f"[Warn] 无 GT: {gt_path}", flush=True)
            continue
        gt_graph = load_cityscale_gt_graph(gt_path)
        gt_graph = relabel_graph_continuous(gt_graph)
        gt_graph = ensure_edge_lengths(gt_graph)

        apls = calc.calculate_apls(pred_graph, gt_graph, canvas_hw=(full_size, full_size))
        topo_p, topo_r, topo_f1 = calc.calculate_topo_precision_recall_f1(
            pred_graph, gt_graph, (full_size, full_size), buffer_px=buf
        )
        elapsed = float(time.perf_counter() - t0)
        pn, pe = pred_graph.number_of_nodes(), pred_graph.number_of_edges()
        gn, ge = gt_graph.number_of_nodes(), gt_graph.number_of_edges()
        apls_list.append(apls)
        topo_p_list.append(topo_p)
        topo_r_list.append(topo_r)
        topo_f1_list.append(topo_f1)
        row = {
            "region": int(t),
            "apls": float(apls),
            "topo_p": float(topo_p),
            "topo_r": float(topo_r),
            "topo_f1": float(topo_f1),
            "pred_nodes": int(pn),
            "pred_edges": int(pe),
            "gt_nodes": int(gn),
            "gt_edges": int(ge),
            "seconds": round(elapsed, 2),
            "index": k,
            "total": n_total,
        }
        per_tile_rows.append(dict(row))
        print(
            f"[{k}/{n_total}] region_{int(t)}  "
            f"pred nodes={pn} edges={pe}  |  GT nodes={gn} edges={ge}  |  "
            f"TOPO-P={topo_p:.4f} TOPO-R={topo_r:.4f} TOPO-F1={topo_f1:.4f} APLS={apls:.4f}  |  {elapsed:.1f}s",
            flush=True,
        )
        if per_tile_f is not None:
            per_tile_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            per_tile_f.flush()

        if graph_dir:
            base = os.path.join(graph_dir, f"region_{int(t)}_pred_graph")
            save_nx_graph_pickle(pred_graph, base + ".pkl")
            save_nx_graph_geojson(pred_graph, base + ".geojson")

    if per_tile_f is not None:
        per_tile_f.close()

    if not apls_list:
        if export_npy_dir:
            print(
                f"[Export] 已写入目录: {export_npy_dir}（无 GT 或未计入指标时不会出现汇总分数）",
                flush=True,
            )
        else:
            print("无有效测试结果。", flush=True)
        return

    mean_apls = float(np.mean(apls_list))
    mean_topo_p = float(np.mean(topo_p_list))
    mean_topo_r = float(np.mean(topo_r_list))
    mean_topo_f1 = float(np.mean(topo_f1_list))
    print("", flush=True)
    print(f"========= City-Scale 拓扑评估  split={args.split}  共 {len(apls_list)} 张 =========", flush=True)
    print("", flush=True)
    print("| # | region | TOPO-P | TOPO-R | TOPO-F1 | APLS | 用时s |", flush=True)
    print("|---:|---:|-----:|-----:|--------:|-----:|------:|", flush=True)
    for i, r in enumerate(per_tile_rows, start=1):
        sec = r.get("seconds", 0.0)
        print(
            f"| {i} | {r['region']} | {r['topo_p']:.4f} | {r['topo_r']:.4f} | "
            f"{r['topo_f1']:.4f} | {r['apls']:.4f} | {sec} |",
            flush=True,
        )
    print("", flush=True)
    print(
        f"均值  TOPO-P={mean_topo_p:.4f}  TOPO-R={mean_topo_r:.4f}  TOPO-F1={mean_topo_f1:.4f}  APLS={mean_apls:.4f}",
        flush=True,
    )
    print("====================================================", flush=True)

    if args.summary_json:
        summ_path = args.summary_json if os.path.isabs(args.summary_json) else os.path.join(PROJECT_ROOT, args.summary_json)
        os.makedirs(os.path.dirname(summ_path) or ".", exist_ok=True)
        payload = {
            "split": args.split,
            "config": cfg_path,
            "checkpoint": g_ckpt,
            "post_threshold": thr,
            "topo_f1_buffer_px": buf,
            "run_date_iso": date.today().isoformat(),
            "mean_apls": mean_apls,
            "mean_topo_p": mean_topo_p,
            "mean_topo_r": mean_topo_r,
            "mean_topo_f1": mean_topo_f1,
            "per_region": per_tile_rows,
        }
        with open(summ_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[Eval] 已写汇总 JSON: {summ_path}", flush=True)


if __name__ == "__main__":
    main()
