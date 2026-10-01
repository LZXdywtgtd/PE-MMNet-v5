# -*- coding: utf-8 -*-
"""V5-030 第六轮：失败模式分类（用户第五节 30min 实验）+ 32×32 oracle 补测。

[A] 32×32 oracle 逐样本通过率补测（第四轮只测了 median chamfer ratio 0.94）
[B] 逐样本失败模式四象限（全部 val 正样本，重点 failed>60px 组）：
      位置口径：逐样本 median pred→裂纹 px
        位置对  <10px（判据 4 要求的 ~10px 级）
        位置近  10~32px（4×4 格粒度内：粗位置对、粒度不够）
        位置远  >32px（连裂纹所在区域都没锁定）
      顺序口径（双指标）：
        主：相邻连线角度差 vs GT kpts（seg_angle_diff，反向容差）
            <30° 对（低于随机 45°） / >=30° 错
        辅：pred 序 vs GT 主轴投影 |Spearman|（zigzag 敏感）
            >=0.9 对 / <0.9 错
    四象限 → v5-β 方向映射（用户表格）：
      位置错+顺序对 → 分辨率/读出；位置对+顺序错 → 顺序监督；
      都错 → 复合；都对 → 阈值/统计
[C] 四象限 × 细度特征交叉（n_px/len/elong/margin）：细度（ρ²≈0.20）之外
    各象限还差什么——回答"80% 未解释方差"
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch
from scipy import stats

from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import catmull_rom_spline_torch

CKPT_BASE = "logs/training_history/v5a6_dprime_resnet18/checkpoints/latest.pt"
VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz"
PX = 128.0
UPPER = 0.0294
REQ = UPPER * 1.5

POS_OK = 10.0    # 位置对阈值 px
POS_FAR = 32.0   # 位置远阈值 px（4×4 格粒度）
ORD_OK = 30.0    # 顺序对阈值（角度差°）
MONO_OK = 0.9    # 顺序对阈值（|Spearman|）


def seg_angle_diff(pred: np.ndarray, gt: np.ndarray) -> float:
    """相邻连线角度差（度），带方向反转容差。与 _diag_kpt_order.py 同口径。"""
    n_seg = min(len(pred), len(gt)) - 1
    if n_seg < 1:
        return float("nan")
    dp = pred[1:n_seg + 1] - pred[:n_seg]
    dg = gt[1:n_seg + 1] - gt[:n_seg]
    np_len = np.linalg.norm(dp, axis=1)
    ng_len = np.linalg.norm(dg, axis=1)
    ok = (np_len > 1e-8) & (ng_len > 1e-8)
    if not ok.any():
        return float("nan")
    dp = dp[ok] / np_len[ok, None]
    dg = dg[ok] / ng_len[ok, None]
    dot = np.clip((dp * dg).sum(axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    ang = np.minimum(ang, 180.0 - ang)
    return float(ang.mean())


def gt_axis(px):
    """GT 像素主轴（质心+第一主成分方向）。"""
    c = px.mean(axis=0)
    c0 = px - c
    cov = c0.T @ c0 / len(px)
    ev, evec = np.linalg.eigh(cov)
    return c, evec[:, np.argmax(ev)]


def main():
    d = np.load(VAL_CACHE)
    px_mask = d["pixel_mask"]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]

    # ---------- [A] oracle 32×32 逐样本补测（+8/16 同口径对照） ----------
    print("[A] oracle 格子中心折线 逐样本通过率（阈值 0.0441）")
    def gt_cells(px, G):
        gy = (px[:, 0] * G).astype(int).clip(0, G - 1)
        gx = (px[:, 1] * G).astype(int).clip(0, G - 1)
        return sorted(set(zip(gy.tolist(), gx.tolist())))

    def pc1_16(centers):
        if len(centers) == 1:
            return centers.repeat(16, axis=0)
        c0 = centers - centers.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(c0, full_matrices=False)
        p = centers[np.argsort(c0 @ vt[0])]
        seg = np.diff(p, axis=0)
        sl = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(sl)])
        s = np.linspace(0, 1, 16) * cum[-1]
        idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(sl) - 1)
        t = ((s - cum[idx]) / np.maximum(sl[idx], 1e-12))[:, None]
        return p[idx] + t * seg[idx]

    def cham(k16, px):
        pk = torch.from_numpy(k16).float().cuda().unsqueeze(0)
        p = torch.from_numpy(px).float().cuda().unsqueeze(0)
        sp = catmull_rom_spline_torch(pk, 200)[0]
        dm = torch.cdist(p[0], sp)
        return (dm.min(dim=-1).values.mean() + dm.min(dim=0).values.mean()).item()

    cnt = {"8": 0, "16": 0, "32": 0}
    n_tot = 0
    for i in pos_idx:
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        if px.shape[0] < 2:
            continue
        n_tot += 1
        for G in (8, 16, 32):
            cells = gt_cells(px, G)
            centers = np.array([[(a + 0.5) / G, (b + 0.5) / G] for a, b in cells])
            if cham(pc1_16(centers), px) <= REQ:
                cnt[str(G)] += 1
    print(f"  oracle  8×8: {cnt['8']}/{n_tot} = {cnt['8']/n_tot*100:.0f}%")
    print(f"  oracle 16×16: {cnt['16']}/{n_tot} = {cnt['16']/n_tot*100:.0f}%")
    print(f"  oracle 32×32: {cnt['32']}/{n_tot} = {cnt['32']/n_tot*100:.0f}%  "
          f"(第四轮仅 median ratio 0.94，此为逐样本补测)")

    # ---------- [B] 模型逐样本：位置 + 双顺序指标 ----------
    ckpt = torch.load(CKPT_BASE, map_location="cuda", weights_only=False)
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
        gt = d["true_keypoints"][i][d["keypoint_mask"][i]].astype(np.float64)
        pr = preds[i].astype(np.float64)
        if px.shape[0] < 2 or len(gt) < 2:
            continue
        kt = torch.from_numpy(pr).float().cuda()
        pt = torch.from_numpy(px).float().cuda()
        med = float(np.median(
            torch.cdist(kt, pt).min(dim=-1).values.cpu().numpy()) * PX)
        ang = seg_angle_diff(pr, gt)
        c, v = gt_axis(px)
        proj = (pr - c) @ v
        if np.std(proj) < 1e-9:
            mono = 0.0
        else:
            mono = abs(stats.spearmanr(np.arange(len(pr)), proj).statistic)
        # 细度特征（与 _diag_bimodal_analysis.py 同口径，统一 px 单位）
        n_px = len(px)
        proj_gt = (px - c) @ v
        len_px = float((proj_gt.max() - proj_gt.min()) * PX)
        w = n_px / max(len_px, 1.0)
        mc = px.mean(axis=0)
        margin = float(min(mc[0], mc[1], 1 - mc[0], 1 - mc[1]) * PX)
        rows.append(dict(idx=i, med=med, ang=ang, mono=mono,
                         n_px=n_px, len_px=len_px, width=w,
                         margin=margin))

    med_arr = np.array([r["med"] for r in rows])
    ang_arr = np.array([r["ang"] for r in rows])
    mono_arr = np.array([r["mono"] for r in rows])

    print(f"\n[B] 逐样本口径（val 正样本 n={len(rows)}）")
    print(f"  位置 med px 分布: p10={np.percentile(med_arr,10):.1f} "
          f"p25={np.percentile(med_arr,25):.1f} p50={np.percentile(med_arr,50):.1f} "
          f"p75={np.percentile(med_arr,75):.1f} p90={np.percentile(med_arr,90):.1f}")
    print(f"  角度差°  分布: p10={np.percentile(ang_arr,10):.1f} "
          f"p25={np.percentile(ang_arr,25):.1f} p50={np.percentile(ang_arr,50):.1f} "
          f"p75={np.percentile(ang_arr,75):.1f} p90={np.percentile(ang_arr,90):.1f}")
    print(f"  单调性|ρ| 分布: p10={np.percentile(mono_arr,10):.2f} "
          f"p50={np.percentile(mono_arr,50):.2f} "
          f"p90={np.percentile(mono_arr,90):.2f}")

    # 校准：GT kpts 自身过这两个指标应≈满分（口径有效性检查）
    cal_ang, cal_mono = [], []
    for i in pos_idx:
        gt = d["true_keypoints"][i][d["keypoint_mask"][i]].astype(np.float64)
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        if px.shape[0] < 2 or len(gt) < 2:
            continue
        cal_ang.append(seg_angle_diff(gt, gt))
        c, v = gt_axis(px)
        pproj = (gt - c) @ v
        cal_mono.append(abs(stats.spearmanr(np.arange(len(gt)), pproj).statistic)
                        if np.std(pproj) >= 1e-9 else 0.0)
    print(f"  口径校准（GT kpts 自测）: 角度差=0 by构造; "
          f"单调性|ρ| p50={np.median(cal_mono):.2f} "
          f">=0.9 占比 {np.mean(np.array(cal_mono) >= MONO_OK)*100:.0f}%"
          f"（若<100% 则 GT 端序与主轴单调亦不完全一致，|ρ| 阈值需降）")

    def quad(r, ord_key):
        pos_ok = r["med"] < POS_OK
        ord_ok = (r["ang"] < ORD_OK) if ord_key == "ang" else (r["mono"] >= MONO_OK)
        if pos_ok and ord_ok:
            return "都对(阈值/统计)"
        if pos_ok and not ord_ok:
            return "位置对+顺序错(顺序监督)"
        if not pos_ok and ord_ok:
            return "位置错+顺序对(分辨率/读出)"
        return "都错(复合)"

    for ord_key, ord_name in (("ang", f"角度差<{ORD_OK:.0f}°"),
                              ("mono", f"|ρ|>={MONO_OK:.1f}")):
        print(f"\n  四象限（位置对=med<{POS_OK:.0f}px；顺序对={ord_name}）")
        labels = ["位置错+顺序对(分辨率/读出)", "位置对+顺序错(顺序监督)",
                  "都错(复合)", "都对(阈值/统计)"]
        for grp_name, grp_mask in (
            ("全部", np.ones(len(rows), bool)),
            ("failed(>60px)", med_arr > 60),
            ("mid(10-60px)", (med_arr >= 10) & (med_arr <= 60)),
            ("learned(<10px)", med_arr < 10),
        ):
            sub = [r for r, m in zip(rows, grp_mask) if m]
            if not sub:
                continue
            cnts = {lb: 0 for lb in labels}
            for r in sub:
                cnts[quad(r, ord_key)] += 1
            n = len(sub)
            parts = "  ".join(
                f"{lb.split('(')[0]}={cnts[lb]}({cnts[lb]/n*100:.0f}%)"
                for lb in labels)
            print(f"    {grp_name:<15} n={n:>3}  {parts}")

    # 位置三档细分（<10 / 10-32 / >32）× 顺序
    print(f"\n  位置三档 × 顺序对（角度差口径）占比:")
    for lo, hi, name in ((0, POS_OK, "<10px"), (POS_OK, POS_FAR, "10-32px"),
                         (POS_FAR, 1e9, ">32px")):
        sub = [r for r in rows if lo <= r["med"] < hi]
        if not sub:
            continue
        ok = sum(1 for r in sub if r["ang"] < ORD_OK)
        mono_ok = sum(1 for r in sub if r["mono"] >= MONO_OK)
        print(f"    位置{name:<8} n={len(sub):>3}  顺序对(角)={ok/len(sub)*100:.0f}%"
              f"  顺序对(单调)={mono_ok/len(sub)*100:.0f}%")

    # ---------- [C] 四象限 × 细度特征 ----------
    print(f"\n[C] 四象限特征交叉（角度差口径；中位数）")
    qmap = {}
    for r in rows:
        qmap.setdefault(quad(r, "ang"), []).append(r)
    feats = ["med", "ang", "mono", "n_px", "len_px", "width", "margin"]
    hdr = "  " + f"{'象限':<26}" + "".join(f"{k:>10}" for k in feats)
    print(hdr)
    for lb in labels:
        sub = qmap.get(lb, [])
        if not sub:
            continue
        vals = "  ".join(
            f"{np.median([r[k] for r in sub]):>10.1f}" for k in feats)
        print(f"  {lb:<26}{vals}")

    # ρ² 口径复核：med_px 与各特征 Spearman
    print("\n  Spearman ρ 与 med_px（ρ² = 单调方差解释比例口径）:")
    for k in ("n_px", "len_px", "width", "margin", "mono", "ang"):
        rho, pv = stats.spearmanr([r[k] for r in rows], med_arr)
        print(f"    {k:<8} ρ={rho:+.3f}  ρ²={rho*rho:.2f}  p={pv:.2e}")

    # 组合解释力：rank 线性回归 R²（top 特征联合能解释多少 med_px 方差）
    X = np.stack([stats.rankdata([r[k] for r in rows])
                  for k in ("n_px", "width", "margin", "len_px")], axis=1)
    y = stats.rankdata(med_arr)
    Xc = np.concatenate([X, np.ones((len(y), 1))], axis=1)
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    r2 = 1 - resid.var() / y.var()
    print(f"    组合（n_px+width+margin+len_px 秩回归）R²={r2:.2f}"
          f"  → 未解释 {(1-r2)*100:.0f}%")

    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
