# -*- coding: utf-8 -*-
"""可视化共享工具：模型加载/推理、样本画像、聚类分簇、oracle chamfer。

所有 _viz_*.py 共用，保证口径与诊断脚本一致（val 197 正样本 +
kd0.5 latest 分组 + 同参聚类）。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from _diag_variance79 import (geom_features, stress_features, GEOM_FEATS,
                              CKPT_BASE, VAL_CACHE, PX, POS_OK, FAIL_TH)
from models import create_v5_model
from training.trainer_v5 import load_checkpoint

CKPT_SPATIAL = "logs/training_history/v5a6_dprime_spatial/checkpoints/latest.pt"
UPPER = 0.0294
REQ = UPPER * 1.5
VIZ_DIR = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz")

SUBTITLE = ("数据来源：val 正样本 197（fast+D'+th60，seed 434242）· "
            "模型 kd0.5 latest 150ep · 判据4上界 0.0294×1.5")


def ensure_viz_dir():
    os.makedirs(VIZ_DIR, exist_ok=True)


def setup_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load_val():
    d = np.load(VAL_CACHE)
    px_mask = d["pixel_mask"]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]
    return d, pos_idx


def load_base_model():
    m = create_v5_model("resnet18", image_channels=3, image_size=128,
                        pretrained_2d=False, use_gp=True, min_kpts=8,
                        max_kpts=16, spatial_head=False)
    load_checkpoint(CKPT_BASE, m, map_location="cuda")
    m.eval().cuda()
    return m


def load_spatial_model():
    m = create_v5_model("resnet18", image_channels=3, image_size=128,
                        pretrained_2d=False, use_gp=True, min_kpts=8,
                        max_kpts=16, spatial_head=True)
    load_checkpoint(CKPT_SPATIAL, m, map_location="cuda")
    m.eval().cuda()
    return m


def infer_base(d, pos_idx=None):
    """kd0.5 baseline 全量推理，返回 (n,16,2) 归一化 kpts。"""
    m = load_base_model()
    n = d["x_2d"].shape[0]
    preds = []
    with torch.no_grad():
        for i in range(0, n, 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            preds.append(m(x1, x2)["keypoints"].cpu().numpy())
    return np.concatenate(preds)


def infer_spatial_heatmap(d):
    """spatial head 150ep 的 heatmap 头输出 (n,1,8,8)——A1 heatmap 格专用。"""
    m = load_spatial_model()
    n = d["x_2d"].shape[0]
    hms = []
    with torch.no_grad():
        for i in range(0, n, 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            hms.append(m(x1, x2)["heatmap"].cpu().numpy())
    return np.concatenate(hms)


def sample_rows(d, pos_idx, preds):
    """逐样本画像（与诊断脚本同口径）：几何+分组。"""
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
    return rows


def cluster_failed(rows):
    """failed 内两簇标签（与 _diag_variance79.py 同参，标签确定性一致）。"""
    failed = [r for r in rows if r["grp"] == "failed"]
    if not failed:
        return {}
    X = np.array([[r[k] for k in GEOM_FEATS] for r in failed], dtype=float)
    ang = np.radians(X[:, GEOM_FEATS.index("dir_deg")])
    X_enc = X[:, [j for j, k in enumerate(GEOM_FEATS) if k != "dir_deg"]]
    X_enc = np.concatenate([X_enc, np.sin(2*ang)[:, None],
                            np.cos(2*ang)[:, None]], axis=1)
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    Xs = StandardScaler().fit_transform(X_enc)
    Xp = PCA(n_components=0.9, random_state=0).fit_transform(Xs)
    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Xp)
    lab = km.labels_
    w = X[:, GEOM_FEATS.index("width")]
    coarse_lab = 0 if np.median(w[lab == 0]) > np.median(w[lab == 1]) else 1
    return {r["idx"]: ("coarse" if lb == coarse_lab else "thin")
            for r, lb in zip(failed, lab)}


def pick_examples(rows, cluster_map, seed=0):
    """四类代表样本：learned-粗 / learned-细 / failed-粗 / failed-细。

    每类取中位代表性（width 最接近组中位数的样本），返回
    [(tag, row), ...]。
    """
    def pick(pool, key="width"):
        if not pool:
            return None
        ws = np.array([r[key] for r in pool])
        return pool[int(np.argsort(np.abs(ws - np.median(ws)))[0])]

    lc = [r for r in rows if r["grp"] == "learned" and r["width"] >= 14]
    lf = [r for r in rows if r["grp"] == "learned" and r["width"] <= 10]
    fc = [r for r in rows if r["grp"] == "failed"
          and cluster_map.get(r["idx"]) == "coarse"]
    ff = [r for r in rows if r["grp"] == "failed"
          and cluster_map.get(r["idx"]) == "thin"]
    tags = (("learned_coarse", "learned-粗(w≥14)", pick(lc)),
            ("learned_fine", "learned-细(w≤10)", pick(lf)),
            ("failed_coarse", "failed-粗(簇0)", pick(fc)),
            ("failed_fine", "failed-细(簇1)", pick(ff)))
    return [(t, label, r) for t, label, r in tags if r is not None]


def oracle_cham(px, G):
    """G×G 格子中心折线（PC1 排序+等弧长 16 点）→ GT 像素 chamfer。"""
    from _diag_v5b_direction2 import cham_of, gt_cells_of, pc1_ordered_16pts
    cells = gt_cells_of(px, G)
    centers = np.array([[(c[0]+0.5)/G, (c[1]+0.5)/G] for c in cells])
    return cham_of(pc1_ordered_16pts(centers), px)


def gt_px(d, i):
    return d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)


def pred_to_crack_px(preds, d, i):
    """逐样本 pred→裂纹像素距离中位数（px）。"""
    px = gt_px(d, i)
    kt = torch.from_numpy(preds[i]).float().cuda()
    pt = torch.from_numpy(px).float().cuda()
    return float(np.median(
        torch.cdist(kt, pt).min(dim=-1).values.cpu().numpy()) * PX)
