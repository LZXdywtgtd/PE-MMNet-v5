# -*- coding: utf-8 -*-
"""位移向量方向诊断：failed 组是"向心收缩"还是"镜像错位"（A1 视觉证据提出）。

背景：A1 四联图显示 idx=13（GT 底→pred 顶）与 idx=176（GT 右→pred 左）
疑似"飘到对侧"，与既往"向心收缩"叙事（pred→心 33px < GT→心 57px）冲突。
既往只测了偏移量没测方向，偏移量对两种机制都成立。

【判据表（事先写死，跑之前定，防解释歧义）】
对 83 个 failed 样本（val 197，kd0.5 latest）统计：
  - 对侧占比 = pred 质心最近边 == GT 质心最近边的对边（上↔下/左↔右）
  - 同边占比 = 两者最近边相同
  - 向心占比 = 单位位移向量 · 单位(GT质心→图心) > 0.5（夹角<60°）
  - 距离保持性 = Spearman(pred最近边距, GT最近边距) 及线性斜率
    （"边界距离"假说直接预测：距离保持、方向错）

判定（互斥）：
  镜像错位成立：对侧占比 > 40% 且 同边占比 < 30%
  向心收缩成立：对侧占比 < 20% 且 向心占比 > 40%
  距离保持加固：Spearman > 0.6 且斜率 ∈ [0.7, 1.3]（与上两条正交，叠加判定）
  其他 → 散乱/混合：两假说均不成立，需新解释

对照：learned/mid 组同表统计（learned 预期同边占主导）；
heatmap 对照（spatial head 4×4 argmax 格子 vs GT 边同边率）验证
"位置信息在模型内已存在，baseline 读出坏了" vs "位置信息整个缺失"。

输出：logs/viz/displacement_mode.json + _viz_displacement_mode.png
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "lib"))

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch

from _viz_common import (ensure_viz_dir, setup_font, load_val, infer_base,
                         sample_rows, gt_px, VIZ_DIR, SUBTITLE,
                         CKPT_SPATIAL, load_spatial_model)
from _diag_variance79 import PX

OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
NAMES = ("上", "下", "左", "右")
QUESTION = ("这张图回答什么问题：failed 组的 pred 整体位移是\"向心收缩\"还是\"镜像错位\""
            "（飘到对侧边）——决定 v5-β 主线是 FPN+位置头 还是 边位置判别头")


def edge_of(gy, gx):
    """质心到四边距离的最小者：0=上 1=下 2=左 3=右（与 _diag_coarse_edge 同口径）。"""
    d = (gy, 127 - gy, gx, 127 - gx)
    return int(np.argmin(d))


def margin_px(gy, gx):
    return float(min(gy, 127 - gy, gx, 127 - gx))


def main():
    ensure_viz_dir()
    plt = setup_font()
    d, pos_idx = load_val()
    preds = infer_base(d)
    rows = sample_rows(d, pos_idx, preds)

    # ---- 逐样本位移量 ----
    recs = []
    for r in rows:
        i = r["idx"]
        px = gt_px(d, i)
        gc = px.mean(axis=0) * PX                    # (y, x)
        pc = preds[i].mean(axis=0) * PX              # (y, x)
        v = pc - gc
        eg, ep = edge_of(gc[0], gc[1]), edge_of(pc[0], pc[1])
        cdir = np.array([64.0, 64.0]) - gc           # GT质心→图心 (y,x)
        nu = np.linalg.norm(v)
        nc = np.linalg.norm(cdir)
        cos_cent = float(np.dot(v, cdir) / (nu * nc)) if nu > 1e-6 and nc > 1e-6 else 0.0
        recs.append(dict(idx=i, grp=r["grp"], edge_gt=eg, edge_pred=ep,
                         opposite=(ep == OPPOSITE[eg]), same=(ep == eg),
                         cos_cent=cos_cent, v=v.tolist(),
                         gt_mg=margin_px(gc[0], gc[1]),
                         pred_mg=margin_px(pc[0], pc[1]),
                         shift=float(np.hypot(*v))))

    # ---- 判据统计（分三组） ----
    summary = {"n": {}, "opposite_pct": {}, "same_pct": {},
               "centripetal_pct": {}, "confusion": {}, "distance_keep": {}}
    from scipy.stats import spearmanr
    for g in ("failed", "mid", "learned"):
        rs = [q for q in recs if q["grp"] == g]
        if not rs:
            continue
        n = len(rs)
        opp = sum(q["opposite"] for q in rs)
        same = sum(q["same"] for q in rs)
        cent = sum(q["cos_cent"] > 0.5 for q in rs)
        summary["n"][g] = n
        summary["opposite_pct"][g] = opp / n * 100
        summary["same_pct"][g] = same / n * 100
        summary["centripetal_pct"][g] = cent / n * 100
        cm = np.zeros((4, 4), dtype=int)
        for q in rs:
            cm[q["edge_gt"], q["edge_pred"]] += 1
        summary["confusion"][g] = cm.tolist()
        gm = np.array([q["gt_mg"] for q in rs])
        pm = np.array([q["pred_mg"] for q in rs])
        rho = float(spearmanr(gm, pm).statistic)
        k = float(np.polyfit(gm, pm, 1)[0]) if gm.std() > 1e-6 else float("nan")
        summary["distance_keep"][g] = {"spearman": rho, "slope": k}
        print(f"[{g}] n={n}  对侧 {opp/n*100:.0f}%  同边 {same/n*100:.0f}%  "
              f"向心(<60°) {cent/n*100:.0f}%  质心偏移 p50={np.median([q['shift'] for q in rs]):.0f}px")
        print(f"      距离保持: Spearman={rho:.3f} 斜率={k:.2f}")
        print("      混淆矩阵(GT行→pred列, 上/下/左/右):")
        for e in range(4):
            print(f"        GT{NAMES[e]}: " + " ".join(f"{cm[e,j]:>3d}" for j in range(4)))

    # ---- 判定（写死判据表） ----
    f_opp = summary["opposite_pct"].get("failed", 0)
    f_same = summary["same_pct"].get("failed", 100)
    f_cent = summary["centripetal_pct"].get("failed", 0)
    dk = summary["distance_keep"].get("failed", {})
    verdicts = []
    if f_opp > 40 and f_same < 30:
        verdicts.append("镜像错位成立")
    if f_opp < 20 and f_cent > 40:
        verdicts.append("向心收缩成立")
    if dk.get("spearman", 0) > 0.6 and 0.7 <= dk.get("slope", 0) <= 1.3:
        verdicts.append("距离保持加固（学到距离没学到方向）")
    if not verdicts:
        verdicts.append("散乱/混合：两假说均不成立，需新解释")
    summary["verdict"] = verdicts
    print(f"\n判定（判据表事先写死）: {'; '.join(verdicts)}")

    # ---- heatmap 对照：spatial head 4×4 argmax 格子边 vs GT 边 ----
    ms = load_spatial_model()
    hm_same = hm_n = 0
    with torch.no_grad():
        for q in recs:
            if q["grp"] != "failed":
                continue
            i = q["idx"]
            x2 = torch.from_numpy(d["x_2d"][i:i+1]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+1]).float().cuda()
            h = ms(x1, x2)["heatmap"][0, 0].cpu().numpy()
            cy, cx = np.unravel_index(np.argmax(h), h.shape)
            if edge_of((cy + 0.5) / 4 * PX, (cx + 0.5) / 4 * PX) == q["edge_gt"]:
                hm_same += 1
            hm_n += 1
    summary["heatmap_same_pct"] = hm_same / max(hm_n, 1) * 100
    print(f"heatmap 对照: spatial 4×4 argmax 格子与 GT 同边 {hm_same}/{hm_n} "
          f"= {summary['heatmap_same_pct']:.0f}%")

    json_path = os.path.join(VIZ_DIR, "displacement_mode.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"落盘 -> {json_path}")

    # ---- 可视化三面板 ----
    frs = [q for q in recs if q["grp"] == "failed"]
    lrs = [q for q in recs if q["grp"] == "learned"]
    fig = plt.figure(figsize=(17, 6))
    fig.suptitle(f"{QUESTION}\n{SUBTITLE}", fontsize=10)

    # 面板1：failed 位移向量玫瑰图（数学角度：右=0° 逆时针，图像 y 向下已翻转）
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    angs = [np.arctan2(-q["v"][0], q["v"][1]) for q in frs]
    mag = np.clip([q["shift"] for q in frs], 5, 130)
    bins = np.linspace(-np.pi, np.pi, 13)
    ax.hist(angs, bins=bins, color="tab:red", alpha=0.75, edgecolor="k")
    ax.set_xticks(np.arange(0, 2*np.pi, np.pi/2))
    ax.set_xticklabels(["右", "上", "左", "下"])
    for a, deg in ((0, "右"), (np.pi/2, "上"), (np.pi, "左"), (-np.pi/2, "下")):
        ax.text(a, 140, deg, fontsize=11, ha="center", color="darkred")
    ax.set_title(f"failed n={len(frs)} 位移向量方向直方图\n"
                 f"(径向=位移长度截断5~130px, GT质心→pred质心)", fontsize=9)

    # 面板2：边混淆矩阵（failed）
    ax = fig.add_subplot(1, 3, 2)
    cm = np.array(summary["confusion"]["failed"], dtype=float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for e in range(4):
        for j in range(4):
            ax.text(j, e, f"{int(cm[e,j])}\n{cmn[e,j]*100:.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if cmn[e, j] > 0.6 else "k")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f"pred{j}" for j in ("上", "下", "左", "右")])
    ax.set_yticklabels([f"GT{j}" for j in ("上", "下", "左", "右")])
    opp_pct = summary["opposite_pct"]["failed"]
    same_pct = summary["same_pct"]["failed"]
    ax.set_title(f"failed 边混淆矩阵（行归一）\n"
                 f"对侧 {opp_pct:.0f}% / 同边 {same_pct:.0f}% / "
                 f"向心(<60°) {f_cent:.0f}%", fontsize=9)

    # 面板3：距离保持散点（"边界距离"假说直接检验）
    ax = fig.add_subplot(1, 3, 3)
    for rs, c, lb in ((lrs, "tab:blue", "learned"), (frs, "tab:red", "failed")):
        ax.scatter([q["gt_mg"] for q in rs], [q["pred_mg"] for q in rs],
                   s=22, c=c, alpha=0.6, label=lb)
    lim = (0, 64)
    ax.plot(lim, lim, "k--", lw=1, label="y=x（距离保持）")
    ax.plot(lim, [127 - v for v in lim], "g:", lw=1,
            label="y=127-x（镜像翻转）")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("GT 质心到最近边距离 px")
    ax.set_ylabel("pred 质心到最近边距离 px")
    sp = summary["distance_keep"]["failed"]
    ax.set_title(f"距离保持性检验（对角线附近=只学到距离）\n"
                 f"failed Spearman={sp['spearman']:.2f} 斜率={sp['slope']:.2f}",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = os.path.join(VIZ_DIR, "_viz_displacement_mode.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"图 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
