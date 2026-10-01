# -*- coding: utf-8 -*-
"""选项 C 补充：failed 组 pred 质心飘到哪里？

_diag_coarse_cluster.py 已证：coarse/thin 两组 failed 的 pred 摊开度
正常（≈GT）但整体飘移（质心偏移 89/101px，贴 mask 率 12%）。
本脚本定位飘移目标：
[P1] pred 质心相对 GT 质心的偏移向量分布（角度/距离直方图）——
     固定角落？对边？随机？
[P2] pred 质心到 σ 峰距离 vs GT 质心到 σ 峰距离——模型是否把
     别的场特征当位置线索；
[P3] pred 质心到图像中心距离——是否向中心/训练均值塌；
[P4] 对照 learned 组同指标（基线水平）。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _diag_variance79 import (geom_features, stress_features, GEOM_FEATS,
                              CKPT_BASE, VAL_CACHE, PX, POS_OK, FAIL_TH)
from models import create_v5_model
from training.trainer_v5 import load_checkpoint


def main():
    d = np.load(VAL_CACHE)
    px_mask = d["pixel_mask"]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]

    model = create_v5_model("resnet18", image_channels=3, image_size=128,
                            pretrained_2d=False, use_gp=True, min_kpts=8,
                            max_kpts=16, spatial_head=False)
    load_checkpoint(CKPT_BASE, model, map_location="cuda")
    model.eval().cuda()
    preds = []
    with torch.no_grad():
        for i in range(0, px_mask.shape[0], 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            preds.append(model(x1, x2)["keypoints"].cpu().numpy())
    preds = np.concatenate(preds)

    rows = []
    for i in pos_idx:
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        if px.shape[0] < 2:
            continue
        kt = torch.from_numpy(preds[i]).float().cuda()
        pt = torch.from_numpy(px).float().cuda()
        med = float(np.median(
            torch.cdist(kt, pt).min(dim=-1).values.cpu().numpy()) * PX)
        g = geom_features(px)
        row = dict(idx=int(i), med=med,
                   grp=("failed" if med > FAIL_TH else
                        "learned" if med < POS_OK else "mid"))
        row.update(g)
        rows.append(row)

    # 聚类同参重拟合分 coarse/thin
    failed = [r for r in rows if r["grp"] == "failed"]
    X = np.array([[r[k] for k in GEOM_FEATS] for r in failed], dtype=float)
    ang = np.radians(X[:, GEOM_FEATS.index("dir_deg")])
    X_enc = X[:, [j for j, k in enumerate(GEOM_FEATS) if k != "dir_deg"]]
    X_enc = np.concatenate([X_enc, np.sin(2*ang)[:, None],
                            np.cos(2*ang)[:, None]], axis=1)
    Xs = StandardScaler().fit_transform(X_enc)
    Xp = PCA(n_components=0.9, random_state=0).fit_transform(Xs)
    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Xp)
    lab = km.labels_
    w = X[:, GEOM_FEATS.index("width")]
    coarse_lab = 0 if np.median(w[lab == 0]) > np.median(w[lab == 1]) else 1
    for j, r in enumerate(failed):
        r["grp"] = "coarse" if lab[j] == coarse_lab else "thin"

    # pred/GT/σ峰 质心
    out = []
    for r in rows:
        i = r["idx"]
        pc = preds[i].mean(axis=0) * PX          # (y,x) px
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        gc = px.mean(axis=0) * PX
        stress = d["x_2d"][i, 1]
        pk = np.unravel_index(np.argmax(stress), stress.shape)
        peak = np.array(pk, dtype=float) + 0.5
        dyx = pc - gc
        out.append(dict(
            grp=r["grp"], med=r["med"], idx=r["idx"],
            offset=float(np.hypot(*dyx)),
            off_ang=float(np.degrees(np.arctan2(dyx[0], dyx[1])) % 360),
            pred2peak=float(np.hypot(*(pc - peak))),
            gt2peak=float(np.hypot(*(gc - peak))),
            pred_cx=pc[1], pred_cy=pc[0],
            pred2center=float(np.hypot(*(pc - 63.5))),
            gt2center=float(np.hypot(*(gc - 63.5))),
        ))
    o = {g: [r for r in out if r["grp"] == g]
         for g in ("coarse", "thin", "learned")}

    for g in ("coarse", "thin", "learned"):
        rs = o[g]
        off = np.array([r["offset"] for r in rs])
        p2p = np.array([r["pred2peak"] for r in rs])
        g2p = np.array([r["gt2peak"] for r in rs])
        c2c = np.array([r["pred2center"] for r in rs])
        g2c = np.array([r["gt2center"] for r in rs])
        print(f"\n[{g}] n={len(rs)}")
        print(f"  质心偏移 px: p50={np.median(off):.1f}")
        print(f"  pred→σ峰 px: p50={np.median(p2p):.1f}   "
              f"GT→σ峰 px: p50={np.median(g2p):.1f}   "
              f"比值 p50={np.median(p2p/np.maximum(g2p,1)):.2f}")
        print(f"  pred→图中心 px: p50={np.median(c2c):.1f}   "
              f"GT→图中心 px: p50={np.median(g2c):.1f}")

    # P1: 偏移角度分布（coarse/thin 合并 failed）——是否集中在某方向
    print("\n[P1] failed 偏移向量角度分布（0=右, 90=下, 顺时针；八区间计数）")
    f_all = [r for r in out if r["grp"] in ("coarse", "thin")]
    angs = np.array([r["off_ang"] for r in f_all])
    hist, _ = np.histogram(angs, bins=8, range=(0, 360))
    for k in range(8):
        bar = "#" * int(hist[k] / max(hist.max(), 1) * 40)
        print(f"  {k*45:>3}-{(k+1)*45:>3}°: {hist[k]:>3} {bar}")
    # 圆均匀性 Rayleigh 检验
    from scipy import stats as st
    rad = np.radians(angs)
    R = np.hypot(np.sin(rad).mean(), np.cos(rad).mean())
    z = len(rad) * R**2
    p_ray = np.exp(-z)
    print(f"  Rayleigh 均匀性: R={R:.3f} z={z:.1f} p≈{p_ray:.2e} "
          f"({'集中' if p_ray < 0.01 else '近似均匀'})")
    # 偏移向量是否指向"GT 的镜像/对边"：偏移后位置是否落在图像对侧
    print("\n[P2] pred 质心落点网格（4×4 计数，failed）")
    grid = np.zeros((4, 4), dtype=int)
    for r in f_all:
        gy = min(int(r["pred_cy"] / 32), 3)
        gx = min(int(r["pred_cx"] / 32), 3)
        grid[gy, gx] += 1
    for gy in range(4):
        print("  " + " ".join(f"{grid[gy, gx]:>3}" for gx in range(4)))
    print("  （对照 GT 质心网格）")
    gridg = np.zeros((4, 4), dtype=int)
    for r in f_all:
        i = r["idx"]
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        gc = px.mean(axis=0) * PX
        gridg[min(int(gc[0] / 32), 3), min(int(gc[1] / 32), 3)] += 1
    for gy in range(4):
        print("  " + " ".join(f"{gridg[gy, gx]:>3}" for gx in range(4)))
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
