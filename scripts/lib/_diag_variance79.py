# -*- coding: utf-8 -*-
"""V5-030 第八轮：79% 未解释方差三合一诊断（v5b_立项依据.md 第四节）。

探针 1：83 个 failed 样本 GT 形态特征 PCA + k-means（k=2..6）+
        silhouette——按判据表（事先写死）判定：
          有结构 = silhouette>0.3 的 k 值 ≥2 个 且 簇间特征差异显著
                   （Mann-Whitney p<0.01 占多数特征）
          无结构 = 最优 k silhouette<0.2 或簇间无显著差异
          弱结构 = 介于两者之间
探针 2：x_2d 应力场空间统计——failed vs learned 逐样本场特征
        （峰值/超阈占比/热点质心位置/梯度/峰-边距离）Mann-Whitney
探针 3：failed vs learned 各抽 10 例应力场可视化并排保存 png
        （探针 2 的定性校验）

口径：val 缓存 + kd0.5 latest 分组（与 _diag_failure_mode.py 一致：
learned=med<10px，failed=med>60px）。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from models import create_v5_model
from training.trainer_v5 import load_checkpoint

CKPT_BASE = "logs/training_history/v5a6_dprime_resnet18/checkpoints/latest.pt"
VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz"
PX = 128.0
POS_OK = 10.0
FAIL_TH = 60.0
TH_NORM = 60.0 / 300.0

GEOM_FEATS = ["n_px", "len_px", "width", "elong", "margin", "dir_deg"]


def geom_features(px):
    """单样本 GT 形态特征（与 _diag_bimodal_analysis.py 同族，px 单位）。"""
    gy = (px[:, 0] * 128).astype(int).clip(0, 127)
    gx = (px[:, 1] * 128).astype(int).clip(0, 127)
    pts = np.stack([gy, gx], axis=1).astype(np.float64)
    c = pts.mean(axis=0)
    c0 = pts - c
    cov = c0.T @ c0 / len(pts)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    proj = c0 @ evec[:, 0]
    len_px = float((proj.max() - proj.min()) * (PX / 128.0))
    w = len(pts) / max(len_px, 1.0)
    ang = np.degrees(np.arctan2(evec[1, 0], evec[0, 0])) % 180.0
    margin = float(min(c[0], c[1], 127 - c[0], 127 - c[1]))
    return dict(n_px=float(len(pts)), len_px=len_px, width=w,
                elong=len_px / max(w, 1.0), margin=margin, dir_deg=ang)


def stress_features(stress):
    """单样本应力场（x_2d 通道1，128×128）空间统计特征。"""
    smax = float(stress.max())
    over = stress > smax * 0.5
    hot = stress > TH_NORM
    ys, xs = np.nonzero(hot)
    if len(ys) > 0:
        cy, cx = float(ys.mean()), float(xs.mean())
        peak_i = np.argmax(stress)
        py, px_ = np.unravel_index(peak_i, stress.shape)
        peak_border = float(min(py, px_, 127 - py, 127 - px_))
    else:
        cy = cx = peak_border = float("nan")
    gy, gx = np.gradient(stress)
    grad = float(np.sqrt(gy**2 + gx**2).mean())
    return dict(s_max=smax, hot_frac=float(hot.mean()),
                half_frac=float(over.mean()), hot_cy=cy, hot_cx=cx,
                hot_std=float(np.std(np.stack([ys, xs], 1)) if len(ys) else np.nan),
                peak_border=peak_border, grad=grad)


def main():
    d = np.load(VAL_CACHE)
    px_mask = d["pixel_mask"]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]

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
        if px.shape[0] < 2:
            continue
        kt = torch.from_numpy(preds[i]).float().cuda()
        pt = torch.from_numpy(px).float().cuda()
        med = float(np.median(
            torch.cdist(kt, pt).min(dim=-1).values.cpu().numpy()) * PX)
        g = geom_features(px)
        s = stress_features(d["x_2d"][i, 1])
        g.update(s)
        g["idx"] = int(i)
        g["med"] = med
        g["grp"] = "failed" if med > FAIL_TH else ("learned" if med < POS_OK else "mid")
        rows.append(g)

    failed = [r for r in rows if r["grp"] == "failed"]
    learned = [r for r in rows if r["grp"] == "learned"]
    print(f"分组: failed={len(failed)} learned={len(learned)} "
          f"mid={sum(1 for r in rows if r['grp']=='mid')}\n")

    # ---------- 探针 1：GT 形态聚类（仅 failed，n=83） ----------
    print("=" * 72)
    print("[探针 1] failed GT 形态聚类（PCA + k-means k=2..6）")
    print("=" * 72)
    X = np.array([[r[k] for k in GEOM_FEATS] for r in failed], dtype=float)
    # dir_deg 是角度（0-180 周期），用 sin/2cos 编码避免 179°≈1° 被判远
    ang = np.radians(X[:, GEOM_FEATS.index("dir_deg")])
    X_enc = X[:, [j for j, k in enumerate(GEOM_FEATS) if k != "dir_deg"]]
    X_enc = np.concatenate([X_enc, np.sin(2 * ang)[:, None],
                            np.cos(2 * ang)[:, None]], axis=1)
    Xs = StandardScaler().fit_transform(X_enc)
    pca = PCA(n_components=0.9, random_state=0)
    Xp = pca.fit_transform(Xs)
    print(f"  特征 {X_enc.shape[1]} 维（角度 sin/cos 编码）→ PCA 保 90% 方差 "
          f"= {Xp.shape[1]} 主成分（各解释率 "
          f"{np.round(pca.explained_variance_ratio_, 2).tolist()}）")

    sil_by_k, labels_by_k = {}, {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xp)
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(Xp, km.labels_))
        sil_by_k[k] = sil
        labels_by_k[k] = km.labels_
    for k, s_ in sil_by_k.items():
        print(f"  k={k}: silhouette={s_:.3f}")

    best_k = max(sil_by_k, key=sil_by_k.get)
    n_k_above = sum(1 for v in sil_by_k.values() if v > 0.3)
    # 簇间差异：best_k 下逐特征 Mann-Whitney（failed 组内两簇）
    lab = labels_by_k[best_k]
    pvals = {}
    for j, kname in enumerate(GEOM_FEATS):
        a = X[:, j][lab == 0]
        b = X[:, j][lab == 1]
        if len(a) >= 3 and len(b) >= 3:
            pvals[kname] = stats.mannwhitneyu(a, b).pvalue
    n_sig = sum(1 for p in pvals.values() if p < 0.01)
    print(f"  最优 k={best_k}（sil={sil_by_k[best_k]:.3f}）；"
          f"sil>0.3 的 k 数={n_k_above}")
    print(f"  best_k 两簇逐特征 Mann-Whitney（簇0 n={int((lab==0).sum())} "
          f"vs 簇1 n={int((lab==1).sum())}）:")
    for kname, p in pvals.items():
        a = X[:, GEOM_FEATS.index(kname)][lab == 0]
        b = X[:, GEOM_FEATS.index(kname)][lab == 1]
        print(f"    {kname:<9} p={p:.2e}  簇0 p50={np.median(a):<8.1f} "
              f"簇1 p50={np.median(b):.1f} {'*' if p < 0.01 else ''}")

    # 按判据表判定（探针 1 单独）
    if n_k_above >= 2 and n_sig >= max(1, len(pvals) // 2):
        probe1 = "有结构"
    elif sil_by_k[best_k] < 0.2 or n_sig == 0:
        probe1 = "无结构"
    else:
        probe1 = "弱结构"
    print(f"  探针 1 单独判定: {probe1}")

    # ---------- 探针 2：应力场空间统计 failed vs learned ----------
    print("\n" + "=" * 72)
    print("[探针 2] x_2d 应力场空间统计 failed vs learned（Mann-Whitney）")
    print("=" * 72)
    sf = ["s_max", "hot_frac", "half_frac", "hot_cy", "hot_cx",
          "hot_std", "peak_border", "grad"]
    farr = {k: np.array([r[k] for r in failed], dtype=float) for k in sf}
    larr = {k: np.array([r[k] for r in learned], dtype=float) for k in sf}
    sig_cnt = 0
    for k in sf:
        a, b = farr[k], larr[k]
        a_ok = a[np.isfinite(a)]
        b_ok = b[np.isfinite(b)]
        if len(a_ok) < 10 or len(b_ok) < 10:
            continue
        p = stats.mannwhitneyu(a_ok, b_ok).pvalue
        sig = p < 0.01
        sig_cnt += int(sig)
        print(f"  {k:<12} failed p50={np.median(a_ok):<8.3f} "
              f"learned p50={np.median(b_ok):<8.3f}  p={p:.2e} "
              f"{'**' if sig else ''}  (n={len(a_ok)}/{len(b_ok)})")
    print(f"  显著特征数（p<0.01）: {sig_cnt}/{len(sf)}")

    # ---------- 探针 3：可视化 ----------
    print("\n[探针 3] 可视化 → _diag_variance79_vis.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    f_sel = rng.choice(len(failed), size=min(10, len(failed)), replace=False)
    l_sel = rng.choice(len(learned), size=min(10, len(learned)), replace=False)
    fig, axes = plt.subplots(8, 5, figsize=(16, 26))
    layout = [(0, "failed", f_sel[:5]), (2, "failed", f_sel[5:]),
              (4, "learned", l_sel[:5]), (6, "learned", l_sel[5:])]
    for row0, grp, sel in layout:
        rlist = failed if grp == "failed" else learned
        for col, si in enumerate(sel):
            r = rlist[int(si)]
            ax = axes[row0, col]
            stress = d["x_2d"][r["idx"], 1]
            im = ax.imshow(stress, cmap="inferno", vmin=0, vmax=0.6)
            ax.set_title(f"{grp} #{r['idx']} med={r['med']:.0f}px", fontsize=8)
            ax.axis("off")
            # GT 裂纹叠加
            px = d["true_crack_pixels"][r["idx"]][d["pixel_mask"][r["idx"]]]
            ax.plot(px[:, 1] * 128, px[:, 0] * 128, "c.", ms=0.5, alpha=0.6)
            ax2 = axes[row0 + 1, col]
            temp = d["x_2d"][r["idx"], 0]
            ax2.imshow(temp, cmap="coolwarm")
            ax2.axis("off")
    fig.suptitle("79% variance probe: stress(top)/temp(bottom), failed(rows1-2) vs learned(rows3-4)")
    fig.tight_layout()
    fig.savefig("_diag_variance79_vis.png", dpi=90)
    plt.close(fig)
    print("  已保存 _diag_variance79_vis.png（上两行 failed、下两行 learned；"
          "每列上=应力场+GT裂纹叠加、下=温度场）")

    # ---------- 汇总判定（按判据表） ----------
    print("\n" + "=" * 72)
    print("[汇总判定]（v5b_立项依据.md 第四节判据表）")
    print("=" * 72)
    if probe1 == "有结构":
        final = "有结构"
    elif probe1 == "无结构":
        # 探针 2/3 佐证修正：场特征显著数多（>=半数）则升弱结构
        final = "弱结构" if sig_cnt >= len(sf) // 2 else "无结构"
    else:
        final = "弱结构"
    print(f"  探针1={probe1}  探针2显著特征={sig_cnt}/{len(sf)}  "
          f"→ 最终判定: **{final}**")
    if final == "有结构":
        print("  → v5-β 需要针对性架构，非通用 FPN")
    elif final == "无结构":
        print("  → 训练动力学问题（loss 平衡/优化器/学习率），"
              "v5-β 第一件事改为训练动力学诊断，比上 FPN 重要得多")
    else:
        print("  → 需要更大的特征集或更多诊断，暂缓架构决策")
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
