# -*- coding: utf-8 -*-
"""V5-030 第五轮：双峰错误分析——"学会 vs 未学会"分层依据（用户指示）。

样本特征（全部从缓存现算）：
  - GT 裂纹像素数 n_px（面积）
  - 裂纹长度 len_px（像素坐标 PCA 主轴投影范围）
  - 伸长率 elong = len/宽（线状度）
  - 主轴方向角度 dir_deg（相对水平，0-180）
  - 质心贴边距离 margin_px（min 到四边，质心）
  - bbox 归一化面积 bbox_area
  - K_gt（关键点数）
  - 应力场（x_2d 通道1）超阈面积占比 + 峰值
  - 温度场（通道0）std

分组：val 上 kd0.5 latest 逐样本 median px：
  learned: <10px   failed: >60px   mid: 其余
统计两组特征 p25/p50/p75 + Spearman 相关 + 简单判决树式的组间差排序。

train(1000) 同口径复算几何特征（无模型），报告特征分布本身 train/val 一致性。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch
from scipy import ndimage, stats

from models import create_v5_model
from training.trainer_v5 import load_checkpoint

CKPT = "logs/training_history/v5a6_dprime_resnet18/checkpoints/latest.pt"
VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz"
TRAIN_CACHE = "logs/sim_cache/sim_cache_p128_n1000_s424242_t6eb66e1b_th60.0.npz"
PX = 128.0
TH = 60.0 / 300.0  # 归一化应力阈值（应力通道量纲假设 0-300 MPa 需验证）


def sample_features(d, i):
    """单样本几何+场特征 dict。"""
    px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)  # (n,2) y,x
    f = {}
    f["n_px"] = len(px)
    if len(px) < 3:
        return None
    # 连通域数（像素栅格）
    grid = np.zeros((128, 128), dtype=bool)
    gy = (px[:, 0] * 128).astype(int).clip(0, 127)
    gx = (px[:, 1] * 128).astype(int).clip(0, 127)
    grid[gy, gx] = True
    lab, n_comp = ndimage.label(grid, structure=np.ones((3, 3)))
    f["n_comp"] = n_comp
    # 最大连通域几何
    sizes = ndimage.sum(grid, lab, range(1, n_comp + 1))
    main = (lab == (np.argmax(sizes) + 1))
    ys, xs = np.nonzero(main)
    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    c = pts.mean(axis=0)
    f["margin_px"] = min(c[0], c[1], 127 - c[0], 127 - c[1])
    c0 = pts - c
    cov = c0.T @ c0 / len(pts)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev = ev[order]
    evec = evec[:, order]
    proj = c0 @ evec[:, 0]
    f["len_px"] = proj.max() - proj.min()
    w = 2 * np.sqrt(ev[1]) if ev[1] > 0 else 1.0
    f["elong"] = f["len_px"] / max(w, 1.0)
    ang = np.degrees(np.arctan2(evec[1, 0], evec[0, 0])) % 180.0
    f["dir_deg"] = ang
    f["horiz"] = min(ang, 180 - ang) < 30 or min(ang, 180 - ang) > 150  # 近水平
    bb = d["true_bbox"][i]
    f["bbox_area"] = float((bb[2] - bb[0]) * (bb[3] - bb[1]))
    f["K_gt"] = int(d["keypoint_mask"][i].sum())
    # 场统计
    stress = d["x_2d"][i, 1]
    f["stress_max"] = float(stress.max())
    f["stress_frac"] = float((stress > stress.max() * 0.5).mean())
    temp = d["x_2d"][i, 0]
    f["temp_std"] = float(temp.std())
    return f


FEATS = ["n_px", "n_comp", "len_px", "elong", "margin_px", "bbox_area",
         "K_gt", "stress_max", "stress_frac", "temp_std"]


def main():
    dv = np.load(VAL_CACHE)
    n_val = dv["pixel_mask"].shape[0]
    pos_idx = np.where(dv["pixel_mask"].sum(axis=1) > 0)[0]

    # 模型推理
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    model = create_v5_model("resnet18", image_channels=3, image_size=128,
                            pretrained_2d=False, use_gp=True, min_kpts=8,
                            max_kpts=16, spatial_head=False)
    load_checkpoint(CKPT, model, map_location="cuda")
    model.eval().cuda()
    preds = []
    with torch.no_grad():
        for i in range(0, n_val, 50):
            x2 = torch.from_numpy(dv["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(dv["x_1d"][i:i+50]).float().cuda()
            preds.append(model(x1, x2)["keypoints"].cpu().numpy())
    preds = np.concatenate(preds)

    # 逐样本 median px + 特征
    rows = []
    for i in pos_idx:
        px = dv["true_crack_pixels"][i][dv["pixel_mask"][i]].astype(np.float64)
        if px.shape[0] < 2:
            continue
        kt = torch.from_numpy(preds[i]).float().cuda()
        pt = torch.from_numpy(px).float().cuda()
        dmin = torch.cdist(kt, pt).min(dim=-1).values.cpu().numpy() * PX
        med = float(np.median(dmin))
        f = sample_features(dv, i)
        if f is None:
            continue
        f["med_px"] = med
        rows.append(f)

    med_arr = np.array([r["med_px"] for r in rows])
    learned = [r for r in rows if r["med_px"] < 10]
    failed = [r for r in rows if r["med_px"] > 60]
    mid = [r for r in rows if 10 <= r["med_px"] <= 60]
    print(f"分组（val, kd0.5 latest, 逐样本 median px）: "
          f"learned(<10px)={len(learned)}  mid={len(mid)}  failed(>60px)={len(failed)}\n")

    print(f"{'特征':<12} {'learned p50':>11} {'failed p50':>11} "
          f"{'比值(f/l)':>9} {'Spearmanρ':>9} {'p值':>10}")
    corr_rows = []
    for k in FEATS:
        lv = np.array([r[k] for r in learned], dtype=float)
        fv = np.array([r[k] for r in failed], dtype=float)
        ratio = (np.median(fv) / np.median(lv)) if np.median(lv) != 0 else float("inf")
        x = np.array([r[k] for r in rows], dtype=float)
        rho, pv = stats.spearmanr(x, med_arr)
        corr_rows.append((k, rho, pv))
        print(f"{k:<12} {np.median(lv):>11.2f} {np.median(fv):>11.2f} "
              f"{ratio:>9.2f} {rho:>9.3f} {pv:>10.2e}")
    # 方向（categorical）：近水平占比对比
    hl = np.mean([r["horiz"] for r in learned])
    hf = np.mean([r["horiz"] for r in failed])
    print(f"{'近水平占比':<11} {hl*100:>10.0f}% {hf*100:>10.0f}%")

    # 排序：|rho| 降序
    corr_sorted = sorted(corr_rows, key=lambda t: -abs(t[1]))
    print("\n|Spearman ρ| 排序（与 med_px 的单调关系）:")
    for k, rho, pv in corr_sorted:
        sig = "**" if pv < 0.01 else ("*" if pv < 0.05 else "  ")
        print(f"  {k:<12} ρ={rho:+.3f} p={pv:.2e} {sig}")

    # 组间差最大的特征（f/l 比值降序）
    print("\n组间 p50 比值排序（failed/learned）:")
    ratios = []
    for k in FEATS:
        lv = np.median([r[k] for r in learned])
        fv = np.median([r[k] for r in failed])
        if lv != 0:
            ratios.append((k, fv / lv))
    ratios.sort(key=lambda t: -abs(t[1]))
    for k, rr in ratios:
        print(f"  {k:<12} failed/learned = {rr:.2f}")

    # train 集特征分布一致性（无模型）
    dt = np.load(TRAIN_CACHE)
    n_train = dt["pixel_mask"].shape[0]
    tpos = np.where(dt["pixel_mask"].sum(axis=1) > 0)[0]
    rng = np.random.default_rng(0)
    sub = rng.choice(tpos, size=min(200, len(tpos)), replace=False)
    trows = [sample_features(dt, int(i)) for i in sub]
    trows = [r for r in trows if r]
    print(f"\ntrain 抽样 {len(trows)} 特征一致性（val p50 / train p50）:")
    for k in FEATS:
        tv = np.median([r[k] for r in trows])
        vv = np.median([r[k] for r in rows])
        print(f"  {k:<12} val={vv:.2f} train={tv:.2f}")

    # 贴边/方向 × 学习率交叉（用户假设表：贴边/方向/长度）
    print("\n假设对照（用户五假设表）:")
    long_th = np.median([r["len_px"] for r in rows])
    for name, mask_fn in (
        ("长裂纹(len>p50)", lambda r: r["len_px"] > long_th),
        ("贴边(margin<12px)", lambda r: r["margin_px"] < 12),
        ("多分量(n_comp>1)", lambda r: r["n_comp"] > 1),
        ("近水平", lambda r: r["horiz"]),
    ):
        grp = [r for r in rows if mask_fn(r)]
        n_learned = sum(1 for r in grp if r["med_px"] < 10)
        rate = n_learned / len(grp) * 100 if grp else float("nan")
        print(f"  {name:<20} n={len(grp):>3}  learned率={rate:.0f}%")
    print(f"  {'全部':<20} n={len(rows):>3}  "
          f"learned率={sum(1 for r in rows if r['med_px'] < 10)/len(rows)*100:.0f}%")

    # n_px 四分位 learned 率梯度 + 等效宽度（细度直接测量）
    npx_arr = np.array([r["n_px"] for r in rows], dtype=float)
    qs = np.percentile(npx_arr, [25, 50, 75])
    print("\nn_px 四分位 learned 率梯度:")
    qmasks = [
        ("Q1(最少像素)", npx_arr <= qs[0]),
        ("Q2", (npx_arr > qs[0]) & (npx_arr <= qs[1])),
        ("Q3", (npx_arr > qs[1]) & (npx_arr <= qs[2])),
        ("Q4(最多像素)", npx_arr > qs[2]),
    ]
    for name, m in qmasks:
        sub = med_arr[m]
        print(f"  {name:<12} n={int(m.sum()):>3}  "
              f"learned(<10px)={np.mean(sub < 10)*100:.0f}%  "
              f"median_px={np.median(sub):.1f}")
    for gname, lo, hi in (("learned(<10px)", 0, 10), ("failed(>60px)", 60, 1e9)):
        grp = [r for r in rows if lo <= r["med_px"] < hi]
        w = [r["n_px"] / max(r["len_px"], 1) for r in grp]
        print(f"{gname}: 等效宽度 n_px/len p50={np.median(w):.1f}px")
    dv.close(); dt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
