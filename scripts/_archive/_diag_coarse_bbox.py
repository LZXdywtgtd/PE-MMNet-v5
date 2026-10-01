# -*- coding: utf-8 -*-
"""选项 C 收尾：bbox 头在 failed 样本上定位对吗？

分流假设：
  bbox 也飘 → 整模空间定位失败（共享特征无位置信息/位置表示瓶颈）；
  bbox 对而 kpts 飘 → 定位信息在，kpt 读出路径独有失败。
口径：val 197，pred bbox vs true_bbox（IoU + 中心距），按组分。
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

from _diag_variance79 import (geom_features, GEOM_FEATS,
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
    preds, bboxes = [], []
    with torch.no_grad():
        for i in range(0, px_mask.shape[0], 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            o = model(x1, x2)
            preds.append(o["keypoints"].cpu().numpy())
            bb = o.get("bbox")
            bboxes.append(bb.cpu().numpy() if bb is not None else None)
    preds = np.concatenate(preds)
    has_bbox = bboxes[0] is not None
    if has_bbox:
        bboxes = np.concatenate(bboxes)
        print(f"bbox 输出形状: {bboxes.shape}")

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
        g.update(dict(idx=int(i), med=med,
                      grp=("failed" if med > FAIL_TH else
                           "learned" if med < POS_OK else "mid")))
        rows.append(g)

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

    if not has_bbox:
        print("模型无 bbox 输出，改用 pred kpts 包络 vs true_bbox")
    for g in ("coarse", "thin", "learned"):
        rs = [r for r in rows if r["grp"] == g]
        ious, cdists = [], []
        for r in rs:
            i = r["idx"]
            if has_bbox:
                pb = bboxes[i]
            else:
                p = preds[i] * PX
                pb = np.array([p[:, 1].min()/PX, p[:, 0].min()/PX,
                               p[:, 1].max()/PX, p[:, 0].max()/PX])
            tb = d["true_bbox"][i]
            ix0, iy0 = max(pb[0], tb[0]), max(pb[1], tb[1])
            ix1, iy1 = min(pb[2], tb[2]), min(pb[3], tb[3])
            inter = max(ix1-ix0, 0) * max(iy1-iy0, 0)
            ap = max(pb[2]-pb[0], 0) * max(pb[3]-pb[1], 0)
            at = max(tb[2]-tb[0], 0) * max(tb[3]-tb[1], 0)
            ious.append(inter / max(ap+at-inter, 1e-6))
            pc = np.array([(pb[0]+pb[2])/2, (pb[1]+pb[3])/2]) * PX
            tc = np.array([(tb[0]+tb[2])/2, (tb[1]+tb[3])/2]) * PX
            cdists.append(float(np.hypot(*(pc-tc))))
        ious, cdists = np.array(ious), np.array(cdists)
        print(f"[{g}] n={len(rs)}  bbox IoU p50={np.median(ious):.3f} "
              f"(IoU>0.5: {(ious>0.5).mean()*100:.0f}%)  "
              f"bbox中心距 p50={np.median(cdists):.1f}px "
              f"(>30px: {(cdists>30).mean()*100:.0f}%)")
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
