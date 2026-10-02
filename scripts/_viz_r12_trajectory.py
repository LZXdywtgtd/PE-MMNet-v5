# -*- coding: utf-8 -*-
"""Round12 工作项 3B（可视化侧）：真梯度下降轨迹投影 + GIF。

输入：round12_param_trajectory/traj_ckpts/ep{N:03d}.pt（3B 训练侧产出）
     round12_loss_landscape/landscape_data.npz（工作项 2 平面）
产出：fig_trajectory.png（轨迹折线叠加 landscape）
     traj_frames/frame_NN.png + traj_anim.gif（PIL 逐帧合成）

投影：平面中心=收官 dp best（全 state），d1/d2=到 s424243/s424245 的
全 state 差向量（与工作项 2/3A 完全同口径，逐点最小二乘投影）。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "lib"))

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch
from PIL import Image

from models import create_v5_model
from training.trainer_v5 import load_checkpoint

from _viz_common import setup_font
setup_font()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = {"blue": "#2a78d6", "orange": "#eb6834", "ink": "#0b0b0b",
     "sec": "#52514e", "mut": "#898781", "grid": "#e1e0d9",
     "axis": "#c3c2b7"}

LAND_OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                        "round12_loss_landscape")
TRAJ_OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                        "round12_param_trajectory")
CKPT_DIR = os.path.join(TRAJ_OUT, "traj_ckpts")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
CHAMP = "v5b_s3_gtfix_dp"
AXIS_RUNS = ["v5b_s4_seed424243", "v5b_s4_seed424245"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEV}")

z = np.load(os.path.join(LAND_OUT, "landscape_data.npz"),
            allow_pickle=False)
a1s, a2s = z["alphas1"], z["alphas2"]
loss_grid = z["loss_grid"]
anchor_runs = [str(s) for s in z["anchor_runs"]]
anchor_coord = z["anchor_coord"]
anchor_loss = z["anchor_loss"]


def build_model():
    return create_v5_model(
        "resnet18", image_channels=3, image_size=128, pretrained_2d=False,
        use_gp=True, min_kpts=8, max_kpts=16, spatial_head=False).to(DEV)


def flat_state(model):
    return np.concatenate(
        [v.detach().double().cpu().numpy().ravel()
         for v in model.state_dict().values()
         if v.dtype.is_floating_point])


def load_flat(path):
    load_checkpoint(path, model, map_location=DEV)
    return flat_state(model)


model = build_model()
theta_c = load_flat(os.path.join(TH, CHAMP, "checkpoints", "best.pt"))
M = np.stack([load_flat(os.path.join(TH, AXIS_RUNS[0], "checkpoints",
                                     "best.pt")) - theta_c,
              load_flat(os.path.join(TH, AXIS_RUNS[1], "checkpoints",
                                     "best.pt")) - theta_c])
Ginv = np.linalg.inv(M @ M.T)

# ---------- 轨迹 checkpoint 投影 ----------
eps = sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pt"))
traj_xy, traj_loss = [], []
with open(os.path.join(TRAJ_OUT, "traj_history.csv"),
          newline="", encoding="utf-8") as f:
    import csv
    hist = list(csv.DictReader(f))
for i, fn in enumerate(eps):
    th = load_flat(os.path.join(CKPT_DIR, fn))
    a = (th - theta_c) @ M.T @ Ginv
    traj_xy.append(a)
    traj_loss.append(float(hist[i]["val_loss"]) if i < len(hist) else np.nan)
    print(f"{fn}  α=({a[0]:+.3f},{a[1]:+.3f})  val={traj_loss[-1]:.4f}")
traj_xy = np.stack(traj_xy)

# 随机初始化位置（轨迹起点附近，作参照）
m0 = build_model()
ri = flat_state(m0)
ri_xy = (ri - theta_c) @ M.T @ Ginv
print(f"随机初始化 α=({ri_xy[0]:+.3f},{ri_xy[1]:+.3f})（轨迹 ep1 起点）")

# ---------- 静态图 ----------
cmap = LinearSegmentedColormap.from_list(
    "loss_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
cmap.set_bad("#e8e7e2")
A1, A2 = np.meshgrid(a1s, a2s, indexing="ij")

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("white")
ax.pcolormesh(A1, A2, loss_grid, cmap=cmap,
              vmin=np.nanmin(loss_grid), vmax=np.nanmax(loss_grid),
              shading="nearest")
ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color=C["axis"], lw=0.8,
        ls=(0, (4, 3)), zorder=2)
ax.plot(traj_xy[:, 0], traj_xy[:, 1], color=C["orange"], lw=2.2,
        marker="o", ms=4.5, zorder=5, label="真梯度下降轨迹（ep1-20）")
ax.scatter(traj_xy[0, 0], traj_xy[0, 1], s=140, marker="^", c=C["ink"],
           zorder=6, edgecolors="white", linewidths=1.5)
ax.annotate(f"ep1\nval={traj_loss[0]:.3f}", traj_xy[0],
            xytext=(12, -30), textcoords="offset points", fontsize=8.5,
            color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.78))
ax.annotate(f"ep{len(eps)}\nval={traj_loss[-1]:.3f}", traj_xy[-1],
            xytext=(-14, 26), textcoords="offset points", fontsize=8.5,
            color=C["ink"], ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.78))
for run, (a1, a2), lv in zip(anchor_runs, anchor_coord, anchor_loss):
    if run == "__random_init__":
        ax.scatter(a1, a2, s=70, c=C["mut"], marker="X", zorder=3,
                   edgecolors="white", linewidths=1.2)
ax.scatter(0, 0, s=110, marker="*", c=C["blue"], zorder=6,
           edgecolors="white", linewidths=1.2)
ax.annotate("dp best（收官）", (0, 0), xytext=(8, 8),
            textcoords="offset points", fontsize=8.5, color=C["blue"])
ax.set_xlabel("α1（→ seed424243 方向）", fontsize=10, color=C["sec"])
ax.set_ylabel("α2（→ seed424245 方向）", fontsize=10, color=C["sec"])
ax.set_title(
    "图A  真梯度下降轨迹：20ep 短重训逐 epoch checkpoint 投影\n"
    "（同口径同 seed 重训；val 逐ep差异≤8.6e-3≈1%，属 GPU 非确定性"
    "良性发散，非逐位复现）",
    fontsize=11.5, color=C["ink"])
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(C["axis"])
ax.tick_params(colors=C["mut"], labelsize=9)
fig.text(0.01, 0.015,
         "注：20ep 为可视化截断（收官 run 150ep）；轨迹是单次训练的真实下降路径，"
         "与图「收敛点分布」的 6 终点分布互补。缓存位级一致（train/val npz 全数组 "
         "identical），loss 差异纯 GPU 核非确定性。",
         fontsize=8, color=C["mut"])
fig.subplots_adjust(bottom=0.12)
pA = os.path.join(TRAJ_OUT, "fig_trajectory.png")
fig.savefig(pA, dpi=110)
plt.close(fig)
print(f"figA -> {pA}")

# ---------- GIF 逐帧 ----------
FRAME_DIR = os.path.join(TRAJ_OUT, "traj_frames")
os.makedirs(FRAME_DIR, exist_ok=True)
frame_paths = []
for i in range(len(eps)):
    fig, ax = plt.subplots(figsize=(8, 6.4))
    fig.patch.set_facecolor("white")
    ax.pcolormesh(A1, A2, loss_grid, cmap=cmap,
                  vmin=np.nanmin(loss_grid), vmax=np.nanmax(loss_grid),
                  shading="nearest")
    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color=C["axis"], lw=0.8,
            ls=(0, (4, 3)), zorder=2)
    ax.plot(traj_xy[:i + 1, 0], traj_xy[:i + 1, 1], color=C["orange"],
            lw=2.2, marker="o", ms=4.5, zorder=5)
    ax.scatter(traj_xy[i, 0], traj_xy[i, 1], s=150, c=C["orange"],
               marker="o", zorder=6, edgecolors="white", linewidths=1.8)
    ax.annotate(f"epoch {i+1}/{len(eps)}   val_loss={traj_loss[i]:.4f}",
                (0.02, 0.97), xycoords="axes fraction", fontsize=11,
                color=C["ink"], va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlim(a1s[0], a1s[-1])
    ax.set_ylim(a2s[0], a2s[-1])
    ax.set_xlabel("α1", fontsize=10, color=C["sec"])
    ax.set_ylabel("α2", fontsize=10, color=C["sec"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["axis"])
    ax.tick_params(colors=C["mut"], labelsize=9)
    fp = os.path.join(FRAME_DIR, f"frame_{i:02d}.png")
    fig.savefig(fp, dpi=90)
    plt.close(fig)
    frame_paths.append(fp)

imgs = [Image.open(p).convert("RGB") for p in frame_paths]
gif_path = os.path.join(TRAJ_OUT, "traj_anim.gif")
imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=450,
             loop=0)
print(f"GIF -> {gif_path}（{len(imgs)} 帧）")
print("DONE round12 工作项3B 可视化侧")
