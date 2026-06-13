"""
图像读写与缩放、路网栅格化、二值膨胀：优先 OpenCV；
若 cv2 缺少 imread/resize/cvtColor/polylines 等（坏安装），回退 PIL + skimage。
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np


def _cv2_module() -> Any:
    return importlib.import_module("cv2")


def cv2_io_ok() -> bool:
    try:
        c = _cv2_module()
        return (
            hasattr(c, "imread")
            and hasattr(c, "resize")
            and hasattr(c, "cvtColor")
            and hasattr(c, "polylines")
            and hasattr(c, "dilate")
        )
    except Exception:
        return False


def cv2_legacy_topo_ok() -> bool:
    """
    仅 ``--legacy-topo-metrics`` 所需：栅格化 + 椭圆膨胀。
    与读图无关（读图可走 PIL）；即使 cv2 无 imread，只要绘图 API 齐全即可用原版 TOPO。
    """
    try:
        c = _cv2_module()
        return (
            hasattr(c, "polylines")
            and hasattr(c, "line")
            and hasattr(c, "getStructuringElement")
            and hasattr(c, "dilate")
            and hasattr(c, "MORPH_ELLIPSE")
        )
    except Exception:
        return False


def rasterize_graph_canvas(graph: Any, hw: tuple[int, int], line_width: int = 3) -> np.ndarray:
    """
    将 NetworkX 图边画到 H×W uint8 画布（与原先 cv2.polylines/line 用途一致）。
    hw 为 (height, width)。
    """
    h, w = int(hw[0]), int(hw[1])
    lw = max(1, int(line_width))
    c = _cv2_module()
    if cv2_io_ok():
        canvas = np.zeros((h, w), dtype=np.uint8)
        for u, v, data in graph.edges(data=True):
            pl = data.get("polyline")
            if pl is not None and len(pl) >= 2:
                pts = np.array([[int(px), int(py)] for px, py in pl], dtype=np.int32)
                c.polylines(canvas, [pts], isClosed=False, color=255, thickness=lw)
            else:
                x0, y0 = graph.nodes[u]["pos"]
                x1, y1 = graph.nodes[v]["pos"]
                c.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), 255, thickness=lw)
        return canvas

    from PIL import Image, ImageDraw

    im = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(im)
    for u, v, data in graph.edges(data=True):
        pl = data.get("polyline")
        if pl is not None and len(pl) >= 2:
            flat = [(int(px), int(py)) for px, py in pl]
            dr.line(flat, fill=255, width=lw)
        else:
            x0, y0 = graph.nodes[u]["pos"]
            x1, y1 = graph.nodes[v]["pos"]
            dr.line([(int(x0), int(y0)), (int(x1), int(y1))], fill=255, width=lw)
    return np.asarray(im, dtype=np.uint8)


def dilate_binary_disk(mask: np.ndarray, buffer_px: int) -> np.ndarray:
    """
    近似原 cv2 椭圆结构元素 (2*buffer_px+1)^2 的膨胀；用 skimage.disk(buffer_px)。
    输入 bool 或 0/1；返回 bool。
    """
    from skimage.morphology import dilation, disk

    m = np.asarray(mask, dtype=bool)
    r = max(0, int(buffer_px))
    if r <= 0:
        return m.copy()
    selem = disk(r)
    try:
        return dilation(m, footprint=selem)
    except TypeError:
        from skimage.morphology import binary_dilation

        return binary_dilation(m, selem=selem)


def imread_gray(path: str) -> np.ndarray | None:
    c = _cv2_module()
    if cv2_io_ok():
        return c.imread(path, c.IMREAD_GRAYSCALE)
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def imread_bgr(path: str) -> np.ndarray | None:
    c = _cv2_module()
    if cv2_io_ok():
        return c.imread(path, c.IMREAD_COLOR)
    from PIL import Image

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return rgb[:, :, ::-1].copy()


def bgr_to_rgb_u8(bgr: np.ndarray) -> np.ndarray:
    c = _cv2_module()
    if cv2_io_ok():
        return c.cvtColor(bgr, c.COLOR_BGR2RGB)
    return bgr[..., ::-1].copy()


def resize(
    img: np.ndarray,
    dsize_wh: tuple[int, int],
    *,
    linear: bool = True,
) -> np.ndarray:
    """
    与 cv2.resize 一致：dsize 为 (width, height)；支持 float 或 uint8 数组。
    """
    c = _cv2_module()
    if cv2_io_ok():
        interp = c.INTER_LINEAR if linear else c.INTER_NEAREST
        return c.resize(img, dsize_wh, interpolation=interp)
    try:
        import skimage.transform as skt
    except ImportError as e:
        raise ImportError(
            "OpenCV 不可用且未安装 skimage，无法 resize。请执行: pip install opencv-python scikit-image"
        ) from e

    ow, oh = int(dsize_wh[0]), int(dsize_wh[1])
    order = 1 if linear else 0
    if img.ndim == 2:
        out = skt.resize(img, (oh, ow), order=order, preserve_range=True, anti_aliasing=linear)
    else:
        out = skt.resize(img, (oh, ow, img.shape[2]), order=order, preserve_range=True, anti_aliasing=linear)
    if img.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out.astype(np.float32)
