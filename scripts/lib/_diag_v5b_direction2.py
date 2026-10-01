# -*- coding: utf-8 -*-
"""V5-030 补充验证：预测 heatmap 格子折线（用户第四节 30min 验证的零训练版）
+ 逐样本通过率稳健性。

[6] baseline 逐样本 chamfer 通过率（判据 4 是 median 口径，通过率看距离多远）
[7] spatial head heatmap h>0.5 格子 → PC1 排序 → 等弧长 16 点 → chamfer
    （argmax+offset 读出方案的"offset=格子中心"下界；对照 GT 格子 oracle 3.89）
[8] oracle 8×8 / 16×16 的逐样本通过率（分辨率结论的稳健性）
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch

from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import catmull_rom_spline_torch

CKPT_SPATIAL = "logs/training_history/v5a6_dprime_spatial/checkpoints/latest.pt"
CKPT_BASE = "logs/training_history/v5a6_dprime_resnet18/checkpoints/latest.pt"
VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz"
PX = 128.0
UPPER = 0.0294
REQ = UPPER * 1.5


def pc1_ordered_16pts(centers):
    if len(centers) == 1:
        return centers.repeat(16, axis=0)
    c0 = centers - centers.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(c0, full_matrices=False)
    order = np.argsort(c0 @ vt[0])
    p = centers[order]
    seg = np.diff(p, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s = np.linspace(0, 1, 16) * cum[-1]
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg_len) - 1)
    t = ((s - cum[idx]) / np.maximum(seg_len[idx], 1e-12))[:, None]
    return p[idx] + t * seg[idx]


def cham_of(kpts16, px):
    pk = torch.from_numpy(kpts16).float().cuda().unsqueeze(0)
    p = torch.from_numpy(px).float().cuda().unsqueeze(0)
    sp = catmull_rom_spline_torch(pk, 200)[0]
    dmat = torch.cdist(p[0], sp)
    return (dmat.min(dim=-1).values.mean() + dmat.min(dim=0).values.mean()).item()


def gt_cells_of(px, G):
    gy = (px[:, 0] * G).astype(int).clip(0, G - 1)
    gx = (px[:, 1] * G).astype(int).clip(0, G - 1)
    return sorted(set(zip(gy.tolist(), gx.tolist())))


def main():
    d = np.load(VAL_CACHE)
    px_mask = d["pixel_mask"]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]

    # ---------- [6][8] oracle/baseline 逐样本通过率 ----------
    print("[6][8] 逐样本 chamfer 通过率（阈值 0.0441 = 上界x1.5）")
    # baseline 推理
    ckpt = torch.load(CKPT_BASE, map_location="cuda", weights_only=False)
    mb = create_v5_model("resnet18", image_channels=3, image_size=128,
                         pretrained_2d=False, use_gp=True, min_kpts=8,
                         max_kpts=16, spatial_head=False)
    load_checkpoint(CKPT_BASE, mb, map_location="cuda")
    mb.eval().cuda()
    preds_b = []
    with torch.no_grad():
        for i in range(0, px_mask.shape[0], 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            preds_b.append(mb(x1, x2)["keypoints"].cpu().numpy())
    preds_b = np.concatenate(preds_b)

    pass_b, pass_o8, pass_o16 = 0, 0, 0
    n_cnt = 0
    for j, i in enumerate(pos_idx):
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        if px.shape[0] < 2:
            continue
        n_cnt += 1
        if cham_of(preds_b[i].astype(np.float64), px) <= REQ:
            pass_b += 1
        for G, cnt in ((8, None), (16, None)):
            cells = gt_cells_of(px, G)
            centers = np.array([[(c[0]+0.5)/G, (c[1]+0.5)/G] for c in cells])
            k16 = pc1_ordered_16pts(centers)
            ch = cham_of(k16, px)
            if G == 8 and ch <= REQ:
                pass_o8 += 1
            if G == 16 and ch <= REQ:
                pass_o16 += 1
    print(f"  baseline(模型) 逐样本通过: {pass_b}/{n_cnt} = {pass_b/n_cnt*100:.0f}%"
          f"  （判据 median 口径需 >50%）")
    print(f"  oracle 8×8 格子中心  逐样本通过: {pass_o8}/{n_cnt} = {pass_o8/n_cnt*100:.0f}%")
    print(f"  oracle 16×16 格子中心 逐样本通过: {pass_o16}/{n_cnt} = {pass_o16/n_cnt*100:.0f}%")

    # ---------- [7] 预测 heatmap 格子折线 ----------
    print("\n[7] spatial head 预测 heatmap → 格子折线（h>0.5，PC1 排序）")
    ckpt = torch.load(CKPT_SPATIAL, map_location="cuda", weights_only=False)
    ms = create_v5_model("resnet18", image_channels=3, image_size=128,
                         pretrained_2d=False, use_gp=True, min_kpts=8,
                         max_kpts=16, spatial_head=True)
    load_checkpoint(CKPT_SPATIAL, ms, map_location="cuda")
    ms.eval().cuda()
    hms = []
    with torch.no_grad():
        for i in range(0, px_mask.shape[0], 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            hms.append(ms(x1, x2)["heatmap"].cpu().numpy())
    hms = np.concatenate(hms)

    chams, n_cells_list, prec_list, rec_list = [], [], [], []
    for i in pos_idx:
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        if px.shape[0] < 2:
            continue
        h = hms[i, 0]
        hot = np.argwhere(h > 0.5)
        if len(hot) == 0:
            hot = np.array([np.unravel_index(np.argmax(h), h.shape)])
        centers = np.array([[(c[0]+0.5)/4, (c[1]+0.5)/4] for c in hot])
        k16 = pc1_ordered_16pts(centers)
        chams.append(cham_of(k16, px))
        n_cells_list.append(len(hot))
        gts = set(gt_cells_of(px, 4))
        ps = set(map(tuple, hot.tolist()))
        prec_list.append(len(gts & ps) / max(len(ps), 1))
        rec_list.append(len(gts & ps) / max(len(gts), 1))
    chams = np.array(chams)
    med = float(np.median(chams))
    print(f"  h>0.5 格子数/样本: p50={np.median(n_cells_list):.0f} "
          f"mean={np.mean(n_cells_list):.1f} (共16格)")
    print(f"  格子级 precision={np.mean(prec_list)*100:.0f}%  "
          f"recall={np.mean(rec_list)*100:.0f}%")
    print(f"  预测格子折线 chamfer p50={med:.4f} ratio={med/UPPER:.2f} "
          f"({'PASS' if med <= REQ else 'FAIL'})  "
          f"逐样本通过 {(chams<=REQ).mean()*100:.0f}%")
    print(f"  （对照 GT 格子 oracle 4×4: ratio 3.89 FAIL——")
    print(f"   若预测≈oracle → 瓶颈确认是 4×4 分辨率而非读出；")
    print(f"   若预测远差于 oracle → 读出/阈值也有问题）")
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
