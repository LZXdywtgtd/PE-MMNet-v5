# -*- coding: utf-8 -*-
"""选项 C 决定性检验：训练集上模型记住了什么？

前两步已证：failed 组（粗/细两簇）pred 摊开度正常但整体飘移
（质心偏移 89/101px），且 pred 向图中心偏（33px vs GT 57px）——
空间定位失败。本脚本回答最后一个分叉：
  若 train 上同簇也失败 → 模型容量/梯度问题（训练动力学）；
  若 train 上同簇学会了   → 空间定位泛化失败（结构问题，架构）。

口径：train 缓存（seed 424242, n=1000）→ 正样本 → 同参推理 →
同参聚类分簇 → 逐组 med/质心偏移。
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
                              CKPT_BASE, PX, POS_OK, FAIL_TH)
from models import create_v5_model
from training.trainer_v5 import load_checkpoint

TRAIN_CACHE = "logs/sim_cache/sim_cache_p128_n1000_s424242_t6eb66e1b_th60.0_dp.npz"


def main():
    d = np.load(TRAIN_CACHE)
    px_mask = d["pixel_mask"]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]
    print(f"train 正样本: {len(pos_idx)}/{px_mask.shape[0]}")

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
        g.update(dict(idx=int(i), med=med,
                      grp=("failed" if med > FAIL_TH else
                           "learned" if med < POS_OK else "mid")))
        rows.append(g)

    n_learned = sum(1 for r in rows if r["grp"] == "learned")
    n_failed = sum(1 for r in rows if r["grp"] == "failed")
    n_mid = sum(1 for r in rows if r["grp"] == "mid")
    print(f"train 分组: learned={n_learned} mid={n_mid} failed={n_failed} "
          f"（failed 率 {n_failed/len(rows)*100:.0f}%，val 为 42%）")

    # failed 内同参聚类分簇
    failed = [r for r in rows if r["grp"] == "failed"]
    if len(failed) >= 10:
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
        nc = int((lab == coarse_lab).sum())
        nt = int((lab != coarse_lab).sum())
        print(f"train failed 两簇: coarse={nc} thin={nt}")

    # 关键：train 上各宽度四分位的 learned 率（是否同宽度也学会了）
    print("\ntrain 按 width 四分位的 learned 率（对照 val 36%→61%）")
    wq = np.percentile([r["width"] for r in rows], [25, 50, 75])
    qs = np.digitize([r["width"] for r in rows], wq)
    for q in range(4):
        sub = [r for r, qq in zip(rows, qs) if qq == q]
        lr = sum(1 for r in sub if r["grp"] == "learned")
        wmed = np.median([r["width"] for r in sub])
        print(f"  Q{q+1} (width p50={wmed:.1f}): learned {lr}/{len(sub)} "
              f"= {lr/len(sub)*100:.0f}%")

    # train failed 的质心偏移（同 val 的飘移模式吗）
    if n_failed:
        offs = []
        for r in rows:
            if r["grp"] != "failed":
                continue
            i = r["idx"]
            pc = preds[i].mean(axis=0) * PX
            px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
            gc = px.mean(axis=0) * PX
            offs.append(float(np.hypot(*(pc - gc))))
        print(f"\ntrain failed 质心偏移 p50={np.median(offs):.1f}px "
              f"（val failed: coarse 89 / thin 101）")
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
