"""
从二值道路掩膜近似生成 IAOF 所需的方向场标签 (H,W,3)：dx, dy, conf。
与 GLD-Road / IAOF 中“沿中心线切向”一致：在骨架像素上用局部邻域估计切向；非道路 conf=0。
"""

from __future__ import annotations

import numpy as np

try:
    from skimage.morphology import skeletonize
except ImportError as e:
    raise ImportError("需要 scikit-image：pip install scikit-image") from e


def _neighbors8(y: int, x: int, h: int, w: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w:
                yield yy, xx


def orientation_from_road_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: HxW uint8/float，道路区域 >0。
    返回 float32 (H,W,3)：channel0=dx, channel1=dy, channel2=conf（道路上为1）。
    方向为单位切向量；骨架外交叉口等处的方向取邻域骨架像素的平均方向。
    """
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    if m.max() == 0:
        return out

    sk = skeletonize(m.astype(bool))
    ys, xs = np.where(sk)
    sk_set = set(zip(ys.tolist(), xs.tolist()))
    if not sk_set:
        return out

    # 骨架像素的切向：指向所有8邻域骨架点的向量和（近似沿线方向）
    tan = np.zeros((h, w, 2), dtype=np.float32)
    for y, x in zip(ys, xs):
        sx = sy = 0.0
        n = 0
        for yy, xx in _neighbors8(y, x, h, w):
            if (yy, xx) in sk_set:
                sx += float(xx - x)
                sy += float(yy - y)
                n += 1
        if n > 0:
            tan[y, x, 0] = sx / n
            tan[y, x, 1] = sy / n

    # 归一化骨架上的切向
    mag = np.sqrt(tan[:, :, 0] ** 2 + tan[:, :, 1] ** 2) + 1e-6
    tan[:, :, 0] /= mag
    tan[:, :, 1] /= mag

    # 道路上非骨架像素：用最近骨架像素的切向（有界 BFS 扩张）
    road = m > 0
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    conf = np.zeros((h, w), dtype=np.float32)

    frontier = [(y, x) for y, x in sk_set]
    dist = np.full((h, w), 1_000_000, dtype=np.int32)
    for y, x in frontier:
        dist[y, x] = 0
        dx[y, x] = tan[y, x, 0]
        dy[y, x] = tan[y, x, 1]
        conf[y, x] = 1.0

    head = 0
    while head < len(frontier):
        y, x = frontier[head]
        head += 1
        d0 = dist[y, x]
        for yy, xx in _neighbors8(y, x, h, w):
            if not road[yy, xx]:
                continue
            if d0 + 1 < dist[yy, xx]:
                dist[yy, xx] = d0 + 1
                dx[yy, xx] = dx[y, x]
                dy[yy, xx] = dy[y, x]
                conf[yy, xx] = 1.0
                frontier.append((yy, xx))

    out[:, :, 0] = dx
    out[:, :, 1] = dy
    out[:, :, 2] = conf
    return out
