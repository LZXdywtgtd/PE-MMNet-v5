# -*- coding: utf-8 -*-
"""Round12 工作项 2（可视化）：loss landscape 热图 + 1D 剖面。

figA_landscape2d.png：2D 插值景观（NaN=发散区灰罩）+ 6 锚点 + 收敛三角
figB_cuts.png：6 条 1D 插值剖面（dp↔各锚点；低-高-低=barrier 曲线）
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
     "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
     "ink": "#0b0b0b", "sec": "#52514e", "mut": "#898781",
     "grid": "#e1e0d9", "axis": "#c3c2b7"}
ANCHOR_C = {"v5b_s3_gtfix_base": C["blue"], "v5b_s4_seed424243": C["orange"],
            "v5b_s4_seed424244": C["aqua"], "v5b_s4_seed424245": C["yellow"],
            "v5b_s4_seed424246": C["magenta"], "__random_init__": C["ink"]}
ANCHOR_LB = {"v5b_s3_gtfix_base": "gtfix_base",
             "v5b_s4_seed424243": "seed424243",
             "v5b_s4_seed424244": "seed424244",
             "v5b_s4_seed424245": "seed424245",
             "v5b_s4_seed424246": "seed424246",
             "__random_init__": "随机初始化"}

OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_loss_landscape")
z = np.load(os.path.join(OUT, "landscape_data.npz"), allow_pickle=False)
a1s, a2s = z["alphas1"], z["alphas2"]
loss = z["loss_grid"]
cut_runs = [str(s) for s in z["cut_runs"]]
cut_t = z["cut_t"]
cut_loss = z["cut_loss"]
anchor_runs = [str(s) for s in z["anchor_runs"]]
anchor_coord = z["anchor_coord"]
anchor_loss = z["anchor_loss"]
center_loss = float(z["center_loss"][0])
swin_loss = float(z["swin_loss"][0])

A1, A2 = np.meshgrid(a1s, a2s, indexing="ij")

# ---------- figA：2D 景观 ----------
# sequential 单色蓝（浅→深=低→高 loss）；发散 NaN 灰罩
vmin = np.nanmin(loss)
vmax = np.nanmax(loss)
cmap = LinearSegmentedColormap.from_list(
    "loss_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
cmap.set_bad("#e8e7e2")

fig, ax = plt.subplots(figsize=(8.6, 7))
fig.patch.set_facecolor("white")
im = ax.pcolormesh(A1, A2, loss, cmap=cmap, vmin=vmin, vmax=vmax,
                   shading="nearest")
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("loss（50 正样本 val，OrderedKeypointLoss total）",
             fontsize=10, color=C["sec"])
cb.ax.tick_params(colors=C["mut"], labelsize=9)

# 收敛三角形（三锚点连线：dp→s424243→s424245→dp）
tri = [(0, 0), (1, 0), (0, 1)]
ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color=C["axis"], lw=0.8,
        ls=(0, (4, 3)), zorder=2)

# 锚点（手动偏移防重叠：seed424244/424246/随机初始化三点密集）
OFF = {"v5b_s3_gtfix_base": (7, 5), "v5b_s4_seed424243": (7, -14),
       "v5b_s4_seed424244": (8, 10), "v5b_s4_seed424245": (7, 5),
       "v5b_s4_seed424246": (-10, -22), "__random_init__": (-6, 12)}
for run, (a1, a2), lv in zip(anchor_runs, anchor_coord, anchor_loss):
    cc = ANCHOR_C[run]
    marker = "X" if run == "__random_init__" else "o"
    ax.scatter(a1, a2, s=90, c=cc, marker=marker, zorder=4,
               edgecolors="white", linewidths=1.5)
    ax.annotate(f"{ANCHOR_LB[run]}\nloss={lv:.3f}", (a1, a2),
                xytext=OFF[run], textcoords="offset points", fontsize=8.5,
                color=C["ink"], zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                          alpha=0.75))

# NaN 区注记
nan_frac = np.isnan(loss).mean()
ax.text(0.02, 0.98,
        f"灰色=发散区（NaN，{nan_frac*100:.0f}%）：景观真实边界，"
        "插值权重物理无意义处前向爆炸",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        color=C["mut"])

ax.set_xlabel("α1（→ seed424243 方向）", fontsize=10, color=C["sec"])
ax.set_ylabel("α2（→ seed424245 方向）", fontsize=10, color=C["sec"])
ax.set_title(
    "图A  loss landscape：收敛三角插值平面\n"
    f"（中心=v5b_s3_gtfix_dp best，端点 loss "
    f"{center_loss:.4f}；网格 21×21，[-0.1,1.1]×[-0.1,1.1]）",
    fontsize=12, color=C["ink"])
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(C["axis"])
ax.tick_params(colors=C["mut"], labelsize=9)
fig.tight_layout()
pA = os.path.join(OUT, "figA_landscape2d.png")
fig.savefig(pA, dpi=110)
plt.close(fig)
print(f"figA -> {pA}")

# ---------- figB：1D 剖面 ----------
fig, ax = plt.subplots(figsize=(9, 5.2))
fig.patch.set_facecolor("white")
cyc = [C["orange"], C["yellow"], C["blue"], C["aqua"], C["magenta"],
       C["ink"]]
for k, (run, ts, ls) in enumerate(zip(cut_runs, cut_t, cut_loss)):
    ax.plot(ts, ls, color=cyc[k % len(cyc)], lw=1.8,
            label=f"dp→{ANCHOR_LB[run]}")
ax.axhline(swin_loss, color=C["green"], lw=1.2, ls=(0, (1, 2)))
ax.annotate(f"swin_yolo 参照 loss={swin_loss:.4f}（不同参数空间，量级参照）",
            (1.24, swin_loss), xytext=(-4, 4), textcoords="offset points",
            fontsize=8, color=C["green"], ha="right")
ax.set_ylim(-0.02, 1.25)
ax.set_xlabel("α（0=dp 收官 best，1=对面锚点 best）", fontsize=10,
              color=C["sec"])
ax.set_ylabel("loss", fontsize=10, color=C["sec"])
ax.set_title(
    "图B  六条线性插值剖面：低—高—低 = 盆地间存在 barrier\n"
    "（dp 与各 seed / 随机初始化之间；NaN 段断线=发散）",
    fontsize=12, color=C["ink"])
ax.grid(True, color=C["grid"], linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(C["axis"])
ax.tick_params(colors=C["mut"], labelsize=9)
ax.legend(fontsize=8.5, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 0.02), ncol=3)
fig.tight_layout()
pB = os.path.join(OUT, "figB_cuts.png")
fig.savefig(pB, dpi=110)
plt.close(fig)
print(f"figB -> {pB}")
print("DONE round12 工作项2 可视化")
