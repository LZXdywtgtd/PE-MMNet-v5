# -*- coding: utf-8 -*-
"""Round12 工作项 1：loss vs epoch 曲线（答辩/论文用）。

figA：5 seed train/val loss 叠加（mean±std 阴影带）+ best epoch 标注
figB：rot90 修复前后（kd0.5=v5a6_dprime_resnet18 vs augfix2）训练动力学对比
figC：收官 run 分项曲线（train 侧 coverage/bbox/gp/poisson 2×2 + 主图）

口径备注：
- best epoch 一律读各 run best.pt 的 epoch 元数据（kd0.5 时代按
  val_loss 选 best，V5-030 后按 coverage_pos 最小化选——checkpoint
  自身记录的是各自时代的口径，CSV argmin(val_coverage) 仅作交叉验证
- v5b_* CSV 14 列 / v5a6_* 13 列——按列名取数
- figB 两 run 均为旧 GT 口径（L 形污染期训练），对比训练动力学非最终精度
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "lib"))

import functools
print = functools.partial(print, flush=True)

import csv
import numpy as np
import torch

from _viz_common import setup_font, VIZ_DIR
import matplotlib
matplotlib.use("Agg")
setup_font()
import matplotlib.pyplot as plt

# dataviz 参考调色板（fix-order）：细线 2px、legend+选择性直标、hairline 网格
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
     "ink": "#0b0b0b", "sec": "#52514e", "mut": "#898781",
     "grid": "#e1e0d9", "axis": "#c3c2b7"}
SEED_C = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"]]

TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_loss_curves")
os.makedirs(OUT, exist_ok=True)

SEED_RUNS = [
    ("v5b_s3_gtfix_dp", "seed 424242"),
    ("v5b_s4_seed424243", "seed 424243"),
    ("v5b_s4_seed424244", "seed 424244"),
    ("v5b_s4_seed424245", "seed 424245"),
    ("v5b_s4_seed424246", "seed 424246"),
]


def read_csv(run):
    path = os.path.join(TH, run, "training_history.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(r[k]) for r in rows])
            for k in rows[0] if k != "timestamp"}
    return cols


def best_epoch(run):
    """best epoch 直接读 best.pt 元数据（其口径随时代：kd0.5 按
    val_loss，V5-030 后按 coverage_pos 最小化）；CSV argmin 交叉验证。"""
    ck = torch.load(os.path.join(TH, run, "checkpoints", "best.pt"),
                    map_location="cpu", weights_only=False)
    ep = int(ck["epoch"])
    d = read_csv(run)
    am = int(d["epoch"][np.argmin(d["val_coverage"])])
    tag = "ok" if am == ep else f"argmin={am}≠ckpt(按val_loss时代)"
    print(f"best_epoch({run}) = ckpt ep{ep} [{tag}]")
    return ep


def style_ax(ax, title, xlabel="epoch", ylabel="loss"):
    ax.set_title(title, fontsize=12, color=C["ink"])
    ax.set_xlabel(xlabel, fontsize=10, color=C["sec"])
    ax.set_ylabel(ylabel, fontsize=10, color=C["sec"])
    ax.grid(True, color=C["grid"], linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["axis"])
    ax.tick_params(colors=C["mut"], labelsize=9)


# ---------- figA：5 seed ----------
all_runs = []
for run, seed in SEED_RUNS:
    d = read_csv(run)
    all_runs.append((run, seed, d, best_epoch(run)))

n = min(len(d["epoch"]) for _, _, d, _ in all_runs)
ep = all_runs[0][2]["epoch"][:n]
tr = np.stack([d["train_loss"][:n] for _, _, d, _ in all_runs])
va = np.stack([d["val_loss"][:n] for _, _, d, _ in all_runs])
beps = [(be, seed) for _, seed, _, be in all_runs]

fig, ax = plt.subplots(figsize=(9, 5.2))
fig.patch.set_facecolor("white")
ax.fill_between(ep, tr.min(0), tr.max(0), color=C["blue"], alpha=0.15,
                label="train [min,max]")
ax.fill_between(ep, va.min(0), va.max(0), color=C["orange"], alpha=0.15,
                label="val [min,max]")
for i, (run, seed, d, _be) in enumerate(all_runs):
    ax.plot(d["epoch"], d["train_loss"], color=SEED_C[i], lw=0.7, alpha=0.45)
    ax.plot(d["epoch"], d["val_loss"], color=SEED_C[i], lw=0.7, alpha=0.45)
ax.plot(ep, tr.mean(0), color=C["blue"], lw=2, label="train loss（5 seed 均值）")
ax.plot(ep, va.mean(0), color=C["orange"], lw=2, label="val loss（5 seed 均值）")
for j, (be, seed) in enumerate(beps):
    ax.axvline(be, color=SEED_C[j], lw=0.8, ls=(0, (4, 3)), alpha=0.7)
    ax.annotate(f"▼ep{be}", (be, ax.get_ylim()[1]), xytext=(2, 8 - 12 * j),
                textcoords="offset points", fontsize=8, color=SEED_C[j],
                ha="left")
ax.set_ylim(0, max(va.max(), 0.9) * 1.15)
style_ax(ax, "图A  干净 GT 重训：5 个训练 seed 的 loss 曲线（细线=单 seed，粗线=均值，阴影=[min,max]）")
ax.legend(fontsize=9, frameon=False, loc="upper right")
fig.tight_layout()
pA = os.path.join(OUT, "figA_5seed_loss.png")
fig.savefig(pA, dpi=110)
plt.close(fig)
print(f"figA -> {pA}")

# ---------- figB：rot90 修复前后 ----------
before = read_csv("v5a6_dprime_resnet18")
after = read_csv("v5a6_dprime_augfix2")
be_before = best_epoch("v5a6_dprime_resnet18")
be_after = best_epoch("v5a6_dprime_augfix2")

fig, ax = plt.subplots(figsize=(9, 5.2))
fig.patch.set_facecolor("white")
ax.plot(before["epoch"], before["train_loss"], color=C["orange"], lw=1.8,
        label="修复前 train_loss（kd0.5）")
ax.plot(before["epoch"], before["val_loss"], color=C["orange"], lw=1.8,
        ls=(0, (4, 3)), label="修复前 val_loss（kd0.5）")
ax.plot(after["epoch"], after["train_loss"], color=C["blue"], lw=1.8,
        label="修复后 train_loss（augfix2）")
ax.plot(after["epoch"], after["val_loss"], color=C["blue"], lw=1.8,
        ls=(0, (4, 3)), label="修复后 val_loss（augfix2）")
for d, cc, be in ((before, C["orange"], be_before),
                  (after, C["blue"], be_after)):
    ax.axvline(be, color=cc, lw=0.8, ls=(0, (1, 2)), alpha=0.7)
    ax.annotate(f"best ep{be}", (be, 0.02), xytext=(3, 0),
                textcoords="offset points", fontsize=8, color=cc)
style_ax(ax, "图B  rot90 增广 bug 修复前后的训练动力学（两 run 均为旧 GT 口径）")
fig.subplots_adjust(bottom=0.18)
fig.text(0.01, 0.015,
         "注：两 run 训练数据均含 L 形边框污染 GT，对比的是增广修复对训练动力学的影响，非最终精度",
         fontsize=8, color=C["mut"])
ax.legend(fontsize=9, frameon=False, loc="upper right")
fig.tight_layout()
pB = os.path.join(OUT, "figB_rot90_before_after.png")
fig.savefig(pB, dpi=110)
plt.close(fig)
print(f"figB -> {pB}")

# ---------- figC：分项曲线 ----------
d = read_csv("v5b_s3_gtfix_dp")
be = best_epoch("v5b_s3_gtfix_dp")
LAM = {"coverage": "λ=1.0", "bbox": "λ=1.0", "gp": "λ=0.1", "poisson": "λ=0.05"}
comp_key = {"coverage": "train_coverage", "bbox": "train_bbox",
            "gp": "train_gp", "poisson": "train_poisson"}
comp_cn = {"coverage": "coverage（样条覆盖率）", "bbox": "bbox（L1）",
           "gp": "gp（高斯过程 LML）", "poisson": "poisson（泊松）"}

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
fig.patch.set_facecolor("white")
for ax0, (k, key) in zip(axes.flat[:4], comp_key.items()):
    ax0.plot(d["epoch"], d[key], color=C["aqua"], lw=1.8)
    ax0.axvline(be, color=C["mut"], lw=0.8, ls=(0, (4, 3)))
    style_ax(ax0, f"{comp_cn[k]}（{LAM[k]}）", ylabel="train loss 分量")
    if k == "gp":
        # LML=(log ℓ)²+(log σ)²，最优即 0；best.pt 实测 2.2e-12，
        # CSV %.6f 恒记 0——线性坐标 + 如实标注，不用 symlog 假动态
        ax0.text(0.98, 0.85, "恒≈0（正则最优即 0，训练后实测 2.2e-12）",
                 transform=ax0.transAxes, ha="right", fontsize=8,
                 color=C["mut"])
for ax0, (key, cn, cc) in zip(
        axes.flat[4:6],
        [("train_loss", "train_loss（加权总损失）", C["blue"]),
         ("val_loss", "val_loss", C["orange"])]):
    ax0.plot(d["epoch"], d[key], color=cc, lw=1.8)
    ax0.axvline(be, color=C["mut"], lw=0.8, ls=(0, (4, 3)))
    style_ax(ax0, cn)
fig.suptitle(
    f"图C  收官 run（v5b_s3_gtfix_dp）loss 分项曲线 ｜ best epoch={be}"
    f"（best.pt 元数据，竖虚线）",
    fontsize=12, color=C["ink"])
fig.tight_layout(rect=(0, 0, 1, 0.95))
pC = os.path.join(OUT, "figC_components.png")
fig.savefig(pC, dpi=110)
plt.close(fig)
print(f"figC -> {pC}")
print("DONE round12 工作项1")
