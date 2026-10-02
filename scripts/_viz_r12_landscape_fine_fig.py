# -*- coding: utf-8 -*-
"""Round12 工作项 2f（可视化）：加密景观渲染。

fig_landscape_fine.png：51×51 [-0.1,0.5]² 加密热图（全域 21×21 版保留
  在 figA_landscape2d.png）；NaN 灰罩 + 锚点。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "lib"))

import functools
print = functools.partial(print, flush=True)

import numpy as np

from _viz_common import setup_font
setup_font()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "magenta": "#e87ba4", "ink": "#0b0b0b",
     "sec": "#52514e", "mut": "#898781", "grid": "#e1e0d9",
     "axis": "#c3c2b7"}
ANCHOR_C = {"v5b_s3_gtfix_base": C["blue"], "v5b_s4_seed424243": C["orange"],
            "v5b_s4_seed424244": C["aqua"], "v5b_s4_seed424245": C["yellow"],
            "v5b_s4_seed424246": C["magenta"], "__random_init__": C["ink"]}
ANCHOR_LB = {"v5b_s3_gtfix_base": "gtfix_base", "v5b_s4_seed424243": "seed424243",
             "v5b_s4_seed424244": "seed424244", "v5b_s4_seed424245": "seed424245",
             "v5b_s4_seed424246": "seed424246", "__random_init__": "随机初始化"}

LAND_OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                        "round12_loss_landscape")
zf = np.load(os.path.join(LAND_OUT, "landscape_data_fine.npz"),
             allow_pickle=False)
z = np.load(os.path.join(LAND_OUT, "landscape_data.npz"), allow_pickle=False)

a1s, a2s = zf["alphas1"], zf["alphas2"]
loss = zf["loss_grid"]
anchor_runs = [str(s) for s in z["anchor_runs"]]
anchor_coord = z["anchor_coord"]
anchor_loss = z["anchor_loss"]
center_loss = float(zf["center_loss"][0])
A1, A2 = np.meshgrid(a1s, a2s, indexing="ij")

cmap = LinearSegmentedColormap.from_list(
    "loss_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
cmap.set_bad("#e8e7e2")

fig, ax = plt.subplots(figsize=(8.6, 7))
fig.patch.set_facecolor("white")
im = ax.pcolormesh(A1, A2, loss, cmap=cmap,
                   vmin=np.nanmin(loss), vmax=np.nanmax(loss),
                   shading="nearest")
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("loss（50 正样本 val，OrderedKeypointLoss total）",
             fontsize=10, color=C["sec"])
cb.ax.tick_params(colors=C["mut"], labelsize=9)

ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color=C["axis"], lw=0.8,
        ls=(0, (4, 3)), zorder=2)

OFF = {"v5b_s3_gtfix_base": (7, 5), "v5b_s4_seed424243": (7, -14),
       "v5b_s4_seed424244": (8, 10), "v5b_s4_seed424245": (7, 5),
       "v5b_s4_seed424246": (-10, -22), "__random_init__": (-6, 12)}
for run, (a1, a2), lv in zip(anchor_runs, anchor_coord, anchor_loss):
    if not (a1s[0] <= a1 <= a1s[-1] and a2s[0] <= a2 <= a2s[-1]):
        continue
    cc = ANCHOR_C[run]
    marker = "X" if run == "__random_init__" else "o"
    ax.scatter(a1, a2, s=90, c=cc, marker=marker, zorder=4,
               edgecolors="white", linewidths=1.5)
    ax.annotate(f"{ANCHOR_LB[run]}\nloss={lv:.3f}", (a1, a2),
                xytext=OFF[run], textcoords="offset points", fontsize=8.5,
                color=C["ink"], zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=0.75))

nan_frac = np.isnan(loss).mean()
ax.text(0.02, 0.98,
        f"灰色=发散区（NaN，{nan_frac*100:.0f}%）；"
        "51×51 加密（[-0.1,0.5]^2，全域版见图A_landscape2d）\n"
        "注：锚点标称 loss=0.015~0.023 为 12M 维原空间直评值，其投影\n"
        "坐标处的面内 loss 本身偏高（离面效应），新旧网格一致非 bug",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        color=C["mut"])

ax.set_xlabel("α1（→ seed424243 方向）", fontsize=10, color=C["sec"])
ax.set_ylabel("α2（→ seed424245 方向）", fontsize=10, color=C["sec"])
ax.set_title(
    "图A-fine  loss landscape 加密扫描：有景观子域 51×51\n"
    f"（中心=v5b_s3_gtfix_dp best，端点 loss {center_loss:.4f}；"
    f"[-0.1,0.5]×[-0.1,0.5]）",
    fontsize=12, color=C["ink"])
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(C["axis"])
ax.tick_params(colors=C["mut"], labelsize=9)
fig.tight_layout()
pOut = os.path.join(LAND_OUT, "fig_landscape_fine.png")
fig.savefig(pOut, dpi=110)
plt.close(fig)
print(f"fig -> {pOut}")
print("DONE round12 工作项2f 可视化")
