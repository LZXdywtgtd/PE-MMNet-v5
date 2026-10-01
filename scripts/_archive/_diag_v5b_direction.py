# -*- coding: utf-8 -*-
"""V5-030 第三轮终局复核（用户质疑 A/B/C + v5-β 方向定量）。

输出五块：
  [1] 分布形状：逐样本 pred→裂纹 px（median/mean 双口径）分位数 +
      点级分布 + 双峰判断——均值 37.4px 到底是"都卡住"还是"混合分布"
  [2] chamfer 分解：fwd(像素→样条) / bwd(样条→像素) 分开——
      回答"spatial head 判据 4 变差(7.06→13.73)的机制"
  [3] 格子级诊断：heatmap argmax/top-k 格子命中率 + 模型 kpts 的
      格子级精度——回答"AUC 0.909 与 37.4px 矛盾"（读出头 vs 格子粗）
  [4] 分辨率上限曲线（核心，无训练）：4/8/16/32 网格"有序格子中心
      折线"平凡预测器的 px 距离与 chamfer ratio——GT 纯计算，
      直接回答 v5-β 需要多高分辨率才能过判据 4（ratio<=1.5 即 ~10px）
  [5] 汇总判读
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch

from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import (
    catmull_rom_spline_torch, coverage_loss,
)

CKPT_SPATIAL = "logs/training_history/v5a6_dprime_spatial/checkpoints/latest.pt"
CKPT_BASE = "logs/training_history/v5a6_dprime_resnet18/checkpoints/latest.pt"
VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz"
PX = 128.0  # patch 像素尺寸（归一化→px 换算）


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    state_keys = ckpt.get("model_state_dict", {}).keys()
    has_spatial = any(k.startswith("spatial_head.") for k in state_keys)
    model = create_v5_model(
        "resnet18", image_channels=3, image_size=128,
        pretrained_2d=False, use_gp=True, min_kpts=8, max_kpts=16,
        spatial_head=has_spatial,
    )
    load_checkpoint(ckpt_path, model, map_location="cuda")
    model.eval().cuda()
    return model, has_spatial


def predict_all(model, d, n_val):
    preds = []
    B = 50
    with torch.no_grad():
        for i in range(0, n_val, B):
            x2 = torch.from_numpy(d["x_2d"][i:i+B]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+B]).float().cuda()
            out = model(x1, x2)
            preds.append(out["keypoints"].cpu().numpy())
    return np.concatenate(preds)  # (N,16,2) (y,x)


def spline_fwd_bwd(pred_kpts_norm, px_norm):
    """单样本：返回 (fwd_px, bwd_px, chamfer_norm)。
    fwd=像素→样条（与训练口径一致 mean+mean），单位 px。"""
    pk = torch.from_numpy(pred_kpts_norm).float().cuda().unsqueeze(0)
    px = torch.from_numpy(px_norm).float().cuda().unsqueeze(0)
    sp = catmull_rom_spline_torch(pk, 200)[0]
    p = px[0]
    dmat = torch.cdist(p, sp)
    fwd = dmat.min(dim=-1).values.mean().item() * PX
    bwd = dmat.min(dim=0).values.mean().item() * PX
    cham = (fwd + bwd) / PX
    return fwd, bwd, cham


def kpt_px_dists(kpts_yx_norm, px_yx_norm):
    """每个 pred 点→最近 GT 像素距离，px。"""
    kt = torch.from_numpy(kpts_yx_norm).float().cuda()
    pt = torch.from_numpy(px_yx_norm).float().cuda()
    d = torch.cdist(kt, pt).min(dim=-1).values.cpu().numpy() * PX
    return d


def ordered_cell_centers(px_yx, G):
    """GT 像素下采样到 G×G 网格，取占用格子中心，沿第一主成分排序，
    等弧长取 16 个点（格子中心折线插值）。返回 (16,2) (y,x) 归一化
    或 None（无占用格子）。"""
    gy = (px_yx[:, 0] * G).astype(int).clip(0, G - 1)
    gx = (px_yx[:, 1] * G).astype(int).clip(0, G - 1)
    cells = sorted(set(zip(gy.tolist(), gx.tolist())))
    if not cells:
        return None
    centers = np.array(
        [[(c[0] + 0.5) / G, (c[1] + 0.5) / G] for c in cells], dtype=np.float64
    )
    if len(centers) == 1:
        base = centers.repeat(16, axis=0)
        return base
    # 第一主成分排序（PC1 投影）
    c0 = centers - centers.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(c0, full_matrices=False)
    proj = c0 @ vt[0]
    order = np.argsort(proj)
    centers = centers[order]
    # 等弧长取 16 点（折线插值，端点含）
    seg = np.diff(centers, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    s = np.linspace(0.0, 1.0, 16) * total
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg_len) - 1)
    t = ((s - cum[idx]) / np.maximum(seg_len[idx], 1e-12))[:, None]
    out = centers[idx] + t * seg[idx]
    return out


def main():
    d = np.load(VAL_CACHE)
    px_mask = d["pixel_mask"]
    n_val = px_mask.shape[0]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]
    print(f"val 正样本: {len(pos_idx)}/{n_val}\n")

    # ---------- [4] 分辨率上限曲线（先算，无模型依赖） ----------
    print("=" * 72)
    print("[4] 分辨率上限曲线：有序格子中心折线平凡预测器（GT 纯计算）")
    print("=" * 72)
    # 判据 4 上界：真 kpts chamfer p50=0.0294 → ratio<=1.5 要求 chamfer<=0.0441
    UPPER = 0.0294
    REQ = UPPER * 1.5
    print(f"判据 4 要求: chamfer <= {REQ:.4f} (上界 0.0294 x 1.5)\n")
    for G in (4, 8, 16, 32):
        pt_dists_all, cham_list = [], []
        n_skip = 0
        for i in pos_idx:
            px = d["true_crack_pixels"][i][d["pixel_mask"][i]]
            if px.shape[0] < 2:
                n_skip += 1
                continue
            ck16 = ordered_cell_centers(px.astype(np.float64), G)
            if ck16 is None:
                n_skip += 1
                continue
            # 每个格子中心→最近 GT 像素
            pt_dists_all.append(kpt_px_dists(ck16, px))
            fwd, bwd, cham = spline_fwd_bwd(ck16, px)
            cham_list.append(cham)
        pt_all = np.concatenate(pt_dists_all)
        cham_arr = np.array(cham_list)
        med_cham = float(np.median(cham_arr))
        print(f"  {G:>2}x{G:<2} 格子({128/G:.0f}px): "
              f"点→裂纹 px p50={np.median(pt_all):5.1f} "
              f"mean={pt_all.mean():5.1f} | "
              f"chamfer p50={med_cham:.4f} ratio={med_cham/UPPER:5.2f} "
              f"{'PASS' if med_cham <= REQ else 'FAIL'}")
    print("  （判读：ratio 随分辨率下降的曲线 = v5-β 需要多高 FPN 的定量答案）")

    # ---------- 模型推理 ----------
    print("\n" + "=" * 72)
    print("[1][2][3] 两个 checkpoint 的分布/分解/格子级诊断")
    print("=" * 72)
    models = {
        "baseline(kd0.5)": (CKPT_BASE, False),
        "spatial(head)": (CKPT_SPATIAL, True),
    }
    summaries = {}
    for name, (path, _) in models.items():
        model, has_sp = load_model(path)
        preds = predict_all(model, d, n_val)
        # heatmap 仅 spatial 有
        hm = None
        if has_sp:
            hms = []
            with torch.no_grad():
                for i in range(0, n_val, 50):
                    x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
                    x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
                    hms.append(model(x1, x2)["heatmap"].cpu().numpy())
            hm = np.concatenate(hms)  # (N,1,H,W)

        per_sample_med, per_sample_mean = [], []
        fwd_all, bwd_all = [], []
        pt_dists_pool = []
        in_gt_cell_cnt, tot_cnt = 0, 0
        argmax_hit, topk_found = [], []
        for j, i in enumerate(pos_idx):
            px = d["true_crack_pixels"][i][d["pixel_mask"][i]]
            if px.shape[0] < 2:
                continue
            pk = preds[i]
            ptd = kpt_px_dists(pk, px)
            pt_dists_pool.append(ptd)
            per_sample_med.append(np.median(ptd))
            per_sample_mean.append(ptd.mean())
            fwd, bwd, _ = spline_fwd_bwd(pk, px)
            fwd_all.append(fwd)
            bwd_all.append(bwd)
            # 格子级：4x4 下 pred 点是否落在 GT 占用格子
            gy = (px[:, 0] * 4).astype(int).clip(0, 3)
            gx = (px[:, 1] * 4).astype(int).clip(0, 3)
            gt_cells = set(zip(gy.tolist(), gx.tolist()))
            pgy = (pk[:, 0] * 4).astype(int).clip(0, 3)
            pgx = (pk[:, 1] * 4).astype(int).clip(0, 3)
            for yy, xx in zip(pgy.tolist(), pgx.tolist()):
                tot_cnt += 1
                if (yy, xx) in gt_cells:
                    in_gt_cell_cnt += 1
            # heatmap argmax/top-k（仅 spatial）
            if hm is not None:
                h = hm[i, 0]
                G_h = h.shape[0]
                tgt_cells = {(yy, xx) for yy, xx in gt_cells}
                am = np.unravel_index(np.argmax(h), h.shape)
                argmax_hit.append(1.0 if (am[0], am[1]) in tgt_cells else 0.0)
                order = np.argsort(h.flatten())[::-1][:16]
                found = sum(
                    1 for f in order
                    if (f // G_h, f % G_h) in tgt_cells
                )
                topk_found.append(found / max(len(tgt_cells), 1))

        pm = np.array(per_sample_med)
        pM = np.array(per_sample_mean)
        pt = np.concatenate(pt_dists_pool)
        print(f"\n----- {name} -----")
        print(f"[1] 逐样本 pred→裂纹 px（median 口径）: "
              f"p10={np.percentile(pm,10):.1f} p25={np.percentile(pm,25):.1f} "
              f"p50={np.percentile(pm,50):.1f} p75={np.percentile(pm,75):.1f} "
              f"p90={np.percentile(pm,90):.1f} mean={pm.mean():.1f}")
        print(f"    逐样本 mean 口径: p50={np.percentile(pM,50):.1f} "
              f"mean={pM.mean():.1f}")
        print(f"    样本 median<10px: {(pm<10).mean()*100:.0f}%  "
              f"<20px: {(pm<20).mean()*100:.0f}%  "
              f">60px: {(pm>60).mean()*100:.0f}%  （双峰则两堆都高）")
        print(f"    点级（16点x样本）px: p10={np.percentile(pt,10):.1f} "
              f"p50={np.percentile(pt,50):.1f} p90={np.percentile(pt,90):.1f} "
              f"<10px 占比 {(pt<10).mean()*100:.0f}%")
        print(f"[2] chamfer 分解(px): fwd(像素→样条) mean={np.mean(fwd_all):.1f} "
              f"p50={np.median(fwd_all):.1f} | bwd(样条→像素) "
              f"mean={np.mean(bwd_all):.1f} p50={np.median(bwd_all):.1f}")
        print(f"    （fwd+bwd 后 ratio: "
              f"{(np.median(np.array(fwd_all)+np.array(bwd_all))/PX)/UPPER:.2f}）")
        print(f"[3] 格子级(4x4): pred 点落入 GT 占用格子 "
              f"{in_gt_cell_cnt/tot_cnt*100:.0f}%（随机基线~裂纹格子占比）")
        if hm is not None:
            print(f"    heatmap: argmax 命中 GT 格子 {np.mean(argmax_hit)*100:.0f}%"
                  f" | top16 找回 GT 格子 {np.mean(topk_found)*100:.0f}%")
        summaries[name] = dict(fwd=np.mean(fwd_all), bwd=np.mean(bwd_all),
                               pm_med=float(np.median(pm)))

    # ---------- [5] 汇总判读 ----------
    print("\n" + "=" * 72)
    print("[5] 汇总判读")
    print("=" * 72)
    b = summaries["baseline(kd0.5)"]
    s = summaries["spatial(head)"]
    print(f"  baseline: fwd={b['fwd']:.1f}px bwd={b['bwd']:.1f}px "
          f"(fwd 主导={b['fwd']>b['bwd']})")
    print(f"  spatial : fwd={s['fwd']:.1f}px bwd={s['bwd']:.1f}px "
          f"(fwd 主导={s['fwd']>s['bwd']})")
    print("  判读指引：spatial bwd 远大于 baseline → 样条 zigzag（顺序乱）"
          "是判据4变差机制，非梯度打架；fwd 相差不大 → 定位本身没变差")
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
