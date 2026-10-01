# -*- coding: utf-8 -*-
"""P0-A1：单样本四联图 ×4（learned-粗 / learned-细 / failed-粗 / failed-细）。

每例 2×2：GT mask+GT kpts / pred kpts+样条 / heatmap 头输出 / x_2d 应力场。
回答问题：模型对四类代表样本分别"看到了什么、画出了什么"。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch

from _viz_common import (ensure_viz_dir, setup_font, load_val, infer_base,
                         sample_rows, cluster_failed, pick_examples,
                         gt_px, VIZ_DIR, SUBTITLE, CKPT_SPATIAL)
from _diag_variance79 import PX
from training.ordered_kp_loss import catmull_rom_spline_torch

QUESTION = ("这张图回答什么问题：模型对四类代表样本（粗/细 × learned/failed）"
            "分别看到了什么输入、画出了什么输出——failed 是\"形状对、位置错\"还是整体崩坏？")


def draw_one(plt, d, preds, hm, row, label):
    i = row["idx"]
    px = gt_px(d, i)
    gt_k = d["true_keypoints"][i][d["keypoint_mask"][i]].astype(np.float64)
    pk = preds[i].astype(np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"{QUESTION}\n{SUBTITLE}", fontsize=9)

    # (0,0) GT mask + GT kpts
    ax = axes[0, 0]
    ax.scatter(px[:, 1] * PX, px[:, 0] * PX, s=4, c="0.6", label="GT 裂纹像素")
    ax.scatter(gt_k[:, 1] * PX, gt_k[:, 0] * PX, s=45, c="tab:blue",
               zorder=3, label=f"GT 关键点 (K={len(gt_k)})")
    ax.set_title("GT mask + GT 关键点（缓存原始值）")
    ax.legend(fontsize=7, loc="upper right")

    # (0,1) pred kpts + 样条（catmull_rom_spline_torch，与 loss 同一实现）
    ax = axes[0, 1]
    kt = torch.from_numpy(pk).float().unsqueeze(0)
    sp = catmull_rom_spline_torch(kt, 200)[0].cpu().numpy()
    ax.scatter(px[:, 1] * PX, px[:, 0] * PX, s=4, c="0.6", label="GT 裂纹像素")
    ax.plot(sp[:, 1] * PX, sp[:, 0] * PX, "r--", lw=1.5, label="pred 样条")
    ax.scatter(pk[:, 1] * PX, pk[:, 0] * PX, s=45, c="red", marker="x",
               zorder=3, label="pred 关键点 (16)")
    ax.set_title(f"pred 关键点+样条 · px误差={row['med']:.1f}px")
    ax.legend(fontsize=7, loc="upper right")

    # (1,0) heatmap 头输出（spatial head）
    ax = axes[1, 0]
    h = hm
    im = ax.imshow(h, cmap="hot", origin="upper",
                   extent=[0, PX, PX, 0], vmin=0, vmax=max(h.max(), 1e-6))
    ax.scatter(px[:, 1] * PX, px[:, 0] * PX, s=4, c="cyan", alpha=0.5,
               label="GT 裂纹像素")
    ax.set_title(f"heatmap 头输出 {h.shape[0]}×{h.shape[1]}"
                 f"（v5a6_dprime_spatial latest）")
    ax.legend(fontsize=7, loc="upper right")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (1,1) x_2d 应力场通道 1
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

    fig.text(0.5, 0.005,
             f"样本 idx={i} · {label} · width={row['width']:.1f}px "
             f"margin={row['margin']:.1f}px len={row['len_px']:.0f}px "
             f"px误差={row['med']:.1f}px",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    out = os.path.join(VIZ_DIR, f"_viz_four_panel_{row['idx']:04d}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    ensure_viz_dir()
    plt = setup_font()
    d, pos_idx = load_val()
    preds = infer_base(d)

    rows = sample_rows(d, pos_idx, preds)
    cmap = cluster_failed(rows)
    picks = pick_examples(rows, cmap)
    if len(picks) < 4:
        print(f"[警告] 只挑出 {len(picks)} 类：{[p[0] for p in picks]}")

    # heatmap 只需 4 例（spatial 模型单独前向）
    from _viz_common import load_spatial_model
    ms = load_spatial_model()
    outs = []
    with torch.no_grad():
        for tag, label, row in picks:
            i = row["idx"]
            x2 = torch.from_numpy(d["x_2d"][i:i+1]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+1]).float().cuda()
            hm = ms(x1, x2)["heatmap"][0, 0].cpu().numpy()
            out = draw_one(plt, d, preds, hm, row, label)
            outs.append(out)
            print(f"  [{tag}] idx={i} {label} -> {out}")
    print("A1 完成：4 张四联图")
    return 0


if __name__ == "__main__":
    sys.exit(main())
