# -*- coding: utf-8 -*-
"""P0-B2：oracle 分辨率曲线（4×4/8×8/16×16/32×32 逐样本通过率）。

回答问题：v5-β 的 FPN 分辨率下限该定在哪——GT 格子中心折线在哪个分辨率
能过判据 4（零训练 oracle 上界）。4 点本次重算并落盘 JSON（4×4 逐样本
通过率既往从未测过）。
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np

from _viz_common import (ensure_viz_dir, setup_font, load_val, gt_px,
                         oracle_cham, VIZ_DIR, SUBTITLE, UPPER, REQ)
from _diag_variance79 import PX

QUESTION = ("这张图回答什么问题：GT 格子中心折线（零训练 oracle 上界）在哪个特征图分辨率\n"
            "能过判据 4（≤0.0441）——v5-β 的 FPN 下限定在哪；32×32 反而下降=PC1 锯齿退化")


def load_or_compute(d, pos_idx):
    """优先读已落盘 JSON（用户规格：数字不重新算）；无则重算并落盘。"""
    json_path = os.path.join(VIZ_DIR, "oracle_pass_rate.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f), json_path, True
    GRID_LIST = (4, 8, 16, 32)
    n_cnt = 0
    pass_cnt = {G: 0 for G in GRID_LIST}
    per_sample = {G: [] for G in GRID_LIST}
    for i in pos_idx:
        px = gt_px(d, i)
        if px.shape[0] < 2:
            continue
        n_cnt += 1
        for G in GRID_LIST:
            ch = oracle_cham(px, G)
            per_sample[G].append(float(ch))
            if ch <= REQ:
                pass_cnt[G] += 1
    summary = {
        "n_samples": n_cnt,
        "req": REQ,
        "upper": UPPER,
        "grids": {str(G): {"pass": pass_cnt[G],
                           "rate": pass_cnt[G] / n_cnt,
                           "chamfer_p50": float(np.median(per_sample[G]))}
                  for G in GRID_LIST},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, json_path, False


def main():
    ensure_viz_dir()
    plt = setup_font()
    d, pos_idx = load_val()

    summary, json_path, from_cache = load_or_compute(d, pos_idx)
    print(f"oracle 数据{'读自缓存' if from_cache else '重算并'}落盘 -> {json_path}")
    n_cnt = summary["n_samples"]
    for G in (4, 8, 16, 32):
        s = summary["grids"][str(G)]
        print(f"  {G}×{G}: 逐样本通过 {s['pass']}/{n_cnt} = {s['rate']*100:.0f}%"
              f"  chamfer p50={s['chamfer_p50']:.4f}")

    # ---- 折线图（数字全部来自 JSON）----
    grids = (4, 8, 16, 32)
    rates = [summary["grids"][str(G)]["rate"] * 100 for G in grids]
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"{QUESTION}\n{SUBTITLE}", fontsize=9)
    ax.plot(grids, rates, "o-", color="tab:blue", lw=2, ms=8)
    for G, r in zip(grids, rates):
        ax.annotate(f"{r:.0f}%", (G, r), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10)
    ax.axhline(80, color="green", ls=":", lw=1.5,
               label="v5-β 验收线 逐样本通过率≥80%（立项文档§五）")
    ax.axhline(59, color="gray", ls="--", lw=1,
               label="oracle 8×8 逐样本 59%（既往实测，交叉校验）")
    ax.annotate("32×32 反而下降\n（PC1 折线退化，oracle 证伪\"更高分辨率更好\"）",
                xy=(32, rates[-1]), xytext=(20, rates[-1] + 22),
                fontsize=9, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred"))
    ax.set_xticks(list(grids))
    ax.set_xticklabels([f"{G}×{G}" for G in grids])
    ax.set_xlabel("特征图分辨率（GT 格子中心折线的格子数）")
    ax.set_ylabel("逐样本通过率 %（chamfer ≤ 0.0441）")
    ax.set_ylim(-5, 112)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(VIZ_DIR, "_viz_oracle_curve.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"B2 完成 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
