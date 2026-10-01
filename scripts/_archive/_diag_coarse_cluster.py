# -*- coding: utf-8 -*-
"""选项 C：粗大型簇（failed 内 n=39，width p50=16.2px 比 learned 均值还宽
却失败）失败机制诊断。

复用 _diag_variance79.py 的聚类（同种子重拟合，标签确定性一致），
把 val 197 分成 coarse_failed / thin_failed / learned / mid 四组，逐组测：

[H1 输入侧]  应力场/温度场在 mask 上 vs 背景的对比度、σ 峰到 mask
             距离——粗大型 failed 的场信号是否比 learned 弱；
[H2 口径伪影] GT 骨架 kpts 等弧长 16 点折线 → GT 像素的 chamfer
             （骨架监督的内在下界）：宽裂纹半宽鸿沟是否让判据 4
             阈值 0.0441 对粗大型内在不可达；
[H3 oracle]  16×16 格子中心折线 chamfer 按簇通过率（分辨率论对
             粗大型是否成立）；
[H4 GT 结构] K_gt / 骨架端点数（分支刺）/ kpts 8px 邻域对 mask 的
             覆盖率 / bbox 实心度——粗大型 GT 结构是否病态；
[H5 pred 行为] pred 质心偏移 / pred 点 20px 内贴 mask 比例 / pred
             摊开度——塌缩、整体偏移还是散乱。

匹配对照：learned 中 width≥14 的"粗型 learned" vs coarse_failed
（同宽度层对照，把宽度本身控住后看还剩什么差异）。
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from _diag_variance79 import (geom_features, stress_features, GEOM_FEATS,
                              CKPT_BASE, VAL_CACHE, PX, POS_OK, FAIL_TH)
from _diag_v5b_direction2 import cham_of, gt_cells_of, pc1_ordered_16pts, REQ, UPPER
from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import resample_kpts_arclength


def field_contrast(stress, temp, px):
    """mask 上 vs 背景的场对比（px 归一化坐标 (y,x)）+ σ 峰到 mask 质心距离。"""
    my = (px[:, 0] * 128).astype(int).clip(0, 127)
    mx = (px[:, 1] * 128).astype(int).clip(0, 127)
    m = np.zeros((128, 128), dtype=bool)
    m[my, mx] = True
    bg = ~m
    s_on, s_bg = float(stress[m].mean()), float(stress[bg].mean())
    t_on, t_bg = float(temp[m].mean()), float(temp[bg].mean())
    peak = np.unravel_index(np.argmax(stress), stress.shape)
    cy, cx = my.mean(), mx.mean()
    peak_d = float(np.hypot(peak[0] - cy, peak[1] - cx))
    return dict(s_on=s_on, s_bg=s_bg, s_contrast=s_on / max(s_bg, 1e-6),
                t_on=t_on, t_bg=t_bg, t_contrast=t_on - t_bg,
                peak_dist=peak_d)


def gt_structure(px, true_kpts, kpt_mask, i):
    """GT 结构：K_gt / 骨架端点数（刺）/ kpts 覆盖率 / bbox 实心度。"""
    from scipy import ndimage
    from skimage import morphology
    gy = (px[:, 0] * 128).astype(int).clip(0, 127)
    gx = (px[:, 1] * 128).astype(int).clip(0, 127)
    m = np.zeros((128, 128), dtype=bool)
    m[gy, gx] = True
    sk = morphology.skeletonize(m)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    nb = ndimage.convolve(sk.astype(np.uint8), kernel, mode="constant") * sk
    n_ep = int((nb == 1).sum())
    # kpts 8px 邻域对 mask 的覆盖率
    ky = (true_kpts[kpt_mask][:, 0] * 128).astype(int).clip(0, 127)
    kx = (true_kpts[kpt_mask][:, 1] * 128).astype(int).clip(0, 127)
    cov = np.zeros((128, 128), dtype=bool)
    for y, x in zip(ky, kx):
        cov[max(0, y-8):y+9, max(0, x-8):x+9] = True
    ys, xs = np.nonzero(m)
    by0, by1 = ys.min(), ys.max()
    bx0, bx1 = xs.min(), xs.max()
    bbox_area = max((by1-by0+1) * (bx1-bx0+1), 1)
    return dict(K_gt=int(kpt_mask.sum()), n_endpoints=n_ep,
                cover8=float(m[cov].mean()),
                solidity=float(len(ys) / bbox_area))


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
        s = stress_features(d["x_2d"][i, 1])
        fc = field_contrast(d["x_2d"][i, 1], d["x_2d"][i, 0], px)
        gs = gt_structure(px, d["true_keypoints"][i], d["keypoint_mask"][i], i)
        # H2: GT 骨架折线内在下界
        tk_raw = d["true_keypoints"][i][d["keypoint_mask"][i]].astype(np.float32)
        tk = torch.from_numpy(tk_raw).cuda().unsqueeze(0)
        km = torch.ones((1, tk_raw.shape[0]), dtype=torch.bool).cuda()
        r16 = resample_kpts_arclength(tk, km, 16)[0]
        floor = cham_of(r16.cpu().numpy().astype(np.float64), px)
        # H3: oracle 16×16
        cells = gt_cells_of(px, 16)
        centers = np.array([[(c[0]+0.5)/16, (c[1]+0.5)/16] for c in cells])
        orc = cham_of(pc1_ordered_16pts(centers), px)
        # H5: pred 行为
        pc = preds[i].mean(axis=0)
        mc = px.mean(axis=0)
        pnn = torch.cdist(kt, pt).min(dim=-1).values.cpu().numpy() * PX
        gt_spread = float(np.median(torch.cdist(pt, pt).cpu().numpy()))
        pd_pair = torch.cdist(kt, kt)
        pred_spread = float(np.median(pd_pair.cpu().numpy()))
        row = dict(idx=int(i), med=med, grp_med=("failed" if med > FAIL_TH
                   else "learned" if med < POS_OK else "mid"),
                   pred_cham=cham_of(preds[i].astype(np.float64), px),
                   floor=floor, oracle16=orc,
                   pred_cdist=float(np.hypot(*(pc - mc)) * PX),
                   pred_within20=float((pnn < 20).mean()),
                   pred_spread=pred_spread, gt_spread=gt_spread)
        row.update(g); row.update(s); row.update(fc); row.update(gs)
        rows.append(row)

    # ---- 聚类（与 _diag_variance79.py 完全同参重拟合） ----
    failed = [r for r in rows if r["grp_med"] == "failed"]
    X = np.array([[r[k] for k in GEOM_FEATS] for r in failed], dtype=float)
    ang = np.radians(X[:, GEOM_FEATS.index("dir_deg")])
    X_enc = X[:, [j for j, k in enumerate(GEOM_FEATS) if k != "dir_deg"]]
    X_enc = np.concatenate([X_enc, np.sin(2*ang)[:, None],
                            np.cos(2*ang)[:, None]], axis=1)
    Xs = StandardScaler().fit_transform(X_enc)
    Xp = PCA(n_components=0.9, random_state=0).fit_transform(Xs)
    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Xp)
    lab = km.labels_
    w0 = np.median(X[lab == 0, GEOM_FEATS.index("width")])
    w1 = np.median(X[lab == 1, GEOM_FEATS.index("width")])
    coarse_lab = 0 if w0 > w1 else 1
    for j, r in enumerate(failed):
        r["grp"] = "coarse_failed" if lab[j] == coarse_lab else "thin_failed"
    for r in rows:
        if "grp" not in r:
            r["grp"] = r["grp_med"] if r["grp_med"] != "failed" else ""
    groups = [g for g in ("coarse_failed", "thin_failed", "learned")
              if any(r["grp"] == g for r in rows)]
    mid = [r for r in rows if r["grp"] == "mid"]

    print(f"分组: coarse_failed={sum(1 for r in rows if r['grp']=='coarse_failed')} "
          f"thin_failed={sum(1 for r in rows if r['grp']=='thin_failed')} "
          f"learned={sum(1 for r in rows if r['grp']=='learned')} mid={len(mid)}")

    def q(arr, ps=(10, 25, 50, 75, 90)):
        return {p: float(np.percentile(arr, p)) for p in ps}

    # ---- 逐组画像 ----
    for g in groups:
        rs = [r for r in rows if r["grp"] == g]
        print("\n" + "=" * 72)
        print(f"[{g}] n={len(rs)}")
        print("=" * 72)
        med = np.array([r["med"] for r in rs])
        print(f"  med px p50={np.median(med):.1f}  "
              f"q={ {k: round(v,1) for k,v in q(med).items()} }")
        for name, key, thr in (
                ("H5 pred chamfer 判据4通过率", "pred_cham", REQ),
                ("H2 GT骨架折线内在下界 达标率(<=REQ)", "floor", REQ),
                ("H3 oracle16 达标率(<=REQ)", "oracle16", REQ)):
            v = np.array([r[key] for r in rs])
            passed = (v <= thr).mean() * 100
            extra = (f"  p50={np.median(v):.4f} ratio={np.median(v)/UPPER:.2f}"
                     if key != "pred_cham" else
                     f"  p50={np.median(v):.4f} ratio={np.median(v)/UPPER:.2f}")
            print(f"  {name}: {passed:.0f}%{extra}")
        f = np.array([r["floor"] for r in rs]) / UPPER
        print(f"  H2 floor/上界 比: p50={np.median(f):.2f} "
              f"p90={np.percentile(f, 90):.2f} (floor/0.0294)")
        for name, key in (("H4 K_gt", "K_gt"), ("H4 骨架端点数", "n_endpoints"),
                          ("H4 kpts8px覆盖率", "cover8"),
                          ("H4 bbox实心度", "solidity"),
                          ("H1 σ_mask对比度(on/bg)", "s_contrast"),
                          ("H1 σ_on", "s_on"), ("H1 σ_bg", "s_bg"),
                          ("H1 T对比(on-bg)", "t_contrast"),
                          ("H1 σ峰→mask质心px", "peak_dist"),
                          ("H5 pred质心偏移px", "pred_cdist"),
                          ("H5 pred贴mask率<20px", "pred_within20"),
                          ("H5 pred摊开度/GT摊开度", None)):
            if key is None:
                a = np.array([r["pred_spread"]/max(r["gt_spread"],1e-6) for r in rs])
            else:
                a = np.array([r[key] for r in rs])
            print(f"  {name}: p50={np.median(a):.2f} "
                  f"q25={np.percentile(a,25):.2f} q75={np.percentile(a,75):.2f}")

    # ---- 匹配对照：coarse_failed vs 粗型 learned（width>=14） ----
    print("\n" + "=" * 72)
    print("[匹配对照] coarse_failed vs 粗型 learned（width≥14px，控住宽度）")
    print("=" * 72)
    cf = [r for r in rows if r["grp"] == "coarse_failed"]
    cl = [r for r in rows if r["grp"] == "learned"
          and r["width"] >= 14.0]
    print(f"  coarse_failed n={len(cf)}  粗型 learned n={len(cl)}"
          f"（learned width p50={np.median([r['width'] for r in cl]):.1f} vs "
          f"coarse_failed p50={np.median([r['width'] for r in cf]):.1f}）")
    feats = [("σ_mask对比度", "s_contrast"), ("σ_on", "s_on"),
             ("T对比(on-bg)", "t_contrast"), ("σ峰→mask质心px", "peak_dist"),
             ("kpts8px覆盖率", "cover8"), ("骨架端点数", "n_endpoints"),
             ("bbox实心度", "solidity"), ("K_gt", "K_gt"),
             ("floor", "floor"), ("oracle16", "oracle16"),
             ("margin", "margin"), ("len_px", "len_px")]
    for name, key in feats:
        a = np.array([r[key] for r in cf], dtype=float)
        b = np.array([r[key] for r in cl], dtype=float)
        p = stats.mannwhitneyu(a, b).pvalue
        star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {name:<14} coarse_failed p50={np.median(a):<9.3f} "
              f"粗型learned p50={np.median(b):<9.3f} p={p:.2e} {star}")

    # ---- 机制小结（数据驱动） ----
    print("\n" + "=" * 72)
    print("[机制小结候选]（按上面数字人工判读）")
    print("=" * 72)
    cf_floor = np.median([r["floor"] for r in cf])
    cl_floor = np.median([r["floor"] for r in cl])
    print(f"  H2 口径伪影: coarse_failed floor p50={cf_floor:.4f} "
          f"(REQ={REQ:.4f}, {'超过→判据4内在不可达' if cf_floor > REQ else '未超→不是主因'}); "
          f"粗型learned floor p50={cl_floor:.4f}")
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
