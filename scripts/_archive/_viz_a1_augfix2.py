# -*- coding: utf-8 -*-
"""A1 四联图重画（augfix2）：同 4 样本（114/27/13/176）同标签，模型换
augfix2 best.pt——修复前后同一样本直接对比。

heatmap 面板仍用旧 spatial checkpoint（增广 bug 时代，已注明）——
该面板的作用是"位置信息在模型内已存在"的历史证据。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch

from _viz_common import (ensure_viz_dir, setup_font, load_val, gt_px, VIZ_DIR,
                         CKPT_SPATIAL, load_spatial_model)
from _diag_variance79 import PX
from _diag_v5b_direction2 import cham_of, UPPER
from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import catmull_rom_spline_torch

CKPT_FIX = "logs/training_history/v5a6_dprime_augfix2/checkpoints/best.pt"
SUBTITLE2 = ("数据来源：val 正样本 197（fast+D'+th60，seed 434242）\n"
             "模型 augfix2 best.pt ep136（rot90 修复重训，判据4 ratio=0.43 通过率 94%）"
             "· 判据4上界 0.0294×1.5")
QUESTION = ("这张图回答什么问题：rot90 增广 bug 修复+正确口径重训后，"
            "原 learned/failed 四类代表样本现在的 pred 落点——failed 组应消失")
# (idx, label)：与旧 A1 完全一致的 4 例
PICKS = ((114, "learned-粗(w≥14)"), (27, "learned-细(w≤10)"),
         (176, "failed-粗(簇0)→修复后"), (13, "failed-细(簇1)→修复后"))


def draw_one(plt, d, preds, hm, i, label, med):
    px = gt_px(d, i)
    gt_k = d["true_keypoints"][i][d["keypoint_mask"][i]].astype(np.float64)
    pk = preds[i].astype(np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"{QUESTION}\n{SUBTITLE2}", fontsize=9)

    ax = axes[0, 0]
    ax.scatter(px[:, 1] * PX, px[:, 0] * PX, s=4, c="0.6", label="GT 裂纹像素")
    ax.scatter(gt_k[:, 1] * PX, gt_k[:, 0] * PX, s=45, c="tab:blue",
               zorder=3, label=f"GT 关键点 (K={len(gt_k)})")
    ax.set_title("GT mask + GT 关键点（缓存原始值）")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[0, 1]
    kt = torch.from_numpy(pk).float().unsqueeze(0)
    sp = catmull_rom_spline_torch(kt, 200)[0].cpu().numpy()
    ax.scatter(px[:, 1] * PX, px[:, 0] * PX, s=4, c="0.6", label="GT 裂纹像素")
    ax.plot(sp[:, 1] * PX, sp[:, 0] * PX, "r--", lw=1.5, label="pred 样条")
    ax.scatter(pk[:, 1] * PX, pk[:, 0] * PX, s=45, c="red", marker="x",
               zorder=3, label="pred 关键点 (16)")
    ax.set_title(f"pred 关键点+样条 · px误差={med:.1f}px")
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1, 0]
    h = hm
    im = ax.imshow(h, cmap="hot", origin="upper",
                   extent=[0, PX, PX, 0], vmin=0, vmax=max(h.max(), 1e-6))
    ax.scatter(px[:, 1] * PX, px[:, 0] * PX, s=4, c="cyan", alpha=0.5,
               label="GT 裂纹像素")
    ax.set_title(f"heatmap 头 {h.shape[0]}×{h.shape[1]}（旧 spatial ckpt）")
    ax.legend(fontsize=7, loc="upper right")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    stress = d["x_2d"][i, 1]
    im = ax.imshow(stress, cmap="inferno", origin="upper",
                   extent=[0, PX, PX, 0], vmin=0, vmax=0.6)
    ax.set_title("输入 x_2d 通道1 应力场")
    plt.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes.flat:
        ax.set_xlim(0, PX)
        ax.set_ylim(PX, 0)
        ax.set_aspect("equal")

    ch = cham_of(pk, px)
    fig.text(0.5, 0.005,
             f"样本 idx={i} · {label} · chamfer={ch:.4f} ({ch*PX:.1f}px) · "
             f"{'PASS' if ch <= UPPER*1.5 else 'FAIL'}",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    out = os.path.join(VIZ_DIR, f"_viz_four_panel_augfix2_{i:04d}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ensure_viz_dir()
    plt = setup_font()
    d, pos_idx = load_val()

    m = create_v5_model("resnet18", image_channels=3, image_size=128,
                        pretrained_2d=False, use_gp=True, min_kpts=8,
                        max_kpts=16, spatial_head=False)
    st = load_checkpoint(CKPT_FIX, m, map_location="cuda")
    print(f"augfix2 best.pt epoch={st.get('epoch')}")
    m.eval().cuda()
    preds = []
    with torch.no_grad():
        for i in range(0, d["x_2d"].shape[0], 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            preds.append(m(x1, x2)["keypoints"].cpu().numpy())
    preds = np.concatenate(preds)

    ms = load_spatial_model()
    outs = []
    with torch.no_grad():
        for i, label in PICKS:
            x2 = torch.from_numpy(d["x_2d"][i:i+1]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+1]).float().cuda()
            hm = ms(x1, x2)["heatmap"][0, 0].cpu().numpy()
            px = gt_px(d, i)
            kt = torch.from_numpy(preds[i]).float().cuda()
            pt = torch.from_numpy(px).float().cuda()
            med = float(np.median(
                torch.cdist(kt, pt).min(dim=-1).values.cpu().numpy()) * PX)
            out = draw_one(plt, d, preds, hm, i, label, med)
            outs.append(out)
            print(f"  idx={i} {label} px误差={med:.1f}px -> {out}")

    # ---- 修复前后对比图（4 行 × 2 列：旧 pred 红 vs 新 pred 绿）----
    base = create_v5_model("resnet18", image_channels=3, image_size=128,
                           pretrained_2d=False, use_gp=True, min_kpts=8,
                           max_kpts=16, spatial_head=False)
    load_checkpoint("logs/training_history/v5a6_dprime_resnet18/checkpoints/latest.pt",
                    base, map_location="cuda")
    base.eval().cuda()
    old_preds = []
    with torch.no_grad():
        for i in range(0, d["x_2d"].shape[0], 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            old_preds.append(base(x1, x2)["keypoints"].cpu().numpy())
    old_preds = np.concatenate(old_preds)

    fig, axes = plt.subplots(4, 2, figsize=(9, 17))
    fig.suptitle(f"rot90 增广 bug 修复前后：同 4 样本 pred 落点对比\n{SUBTITLE2}",
                 fontsize=10)
    for r, (i, label) in enumerate(PICKS):
        px = gt_px(d, i)
        for c, (pp, tag, col) in enumerate(
                ((old_preds[i], "旧 kd0.5（bug 增广）", "red"),
                 (preds[i], "augfix2 best（修复）", "tab:green"))):
            ax = axes[r, c]
            ax.scatter(px[:, 1] * PX, px[:, 0] * PX, s=4, c="0.6",
                       label="GT 裂纹像素")
            ax.scatter(pp[:, 1] * PX, pp[:, 0] * PX, s=40, c=col, marker="x",
                       zorder=3, label=f"{tag} pred")
            ch = cham_of(pp.astype(np.float64), px)
            ax.set_title(f"idx={i} {label}\n{tag} · chamfer={ch:.4f} ({ch*PX:.1f}px)",
                         fontsize=9)
            ax.set_xlim(0, PX); ax.set_ylim(PX, 0); ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(VIZ_DIR, "_viz_augfix2_before_after.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"对比图 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
