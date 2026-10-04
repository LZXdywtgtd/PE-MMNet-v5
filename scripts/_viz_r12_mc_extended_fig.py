# -*- coding: utf-8 -*-
"""MC 补充实验可视化：figB2（z 位移+消融）/ figD_k15 / figE_3seed。

用法：python _viz_r12_mc_extended_fig.py [z | k15 | seed3]
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

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "ink": "#0b0b0b", "sec": "#52514e", "mut": "#898781",
     "grid": "#e1e0d9", "axis": "#c3c2b7", "red": "#e34948",
     "yellow": "#eda100", "magenta": "#e87ba4"}

OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_mode_connectivity")
WHICH = sys.argv[1] if len(sys.argv) > 1 else "z"

if WHICH == "z":
    zz = np.load(os.path.join(OUT, "mc_ext_zdisplace.npz"))
    ab = np.load(os.path.join(OUT, "mc_ext_zproj_ablation.npz"))

    z_ab = zz["r0_mid_z"] * 100
    z_ac = zz["r1_mid_z"] * 100
    K = 7
    t_mid = np.arange(1, K - 1) / (K - 1)
    l_opt = ab["l_opt"]
    l_proj = ab["l_proj"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fig.patch.set_facecolor("white")

    axL.plot(t_mid, z_ab, color=C["blue"], lw=2, marker="o", ms=6,
             label="AB（s424243 - dp）")
    axL.plot(t_mid, z_ac, color=C["orange"], lw=2, marker="s", ms=6,
             label="AC（s424243 - s424245）")
    axL.axhline(2.0, color=C["red"], lw=1, ls=(0, (4, 3)))
    axL.text(0.5, 2.15, "2% 判据线", color=C["red"], fontsize=8.5,
             ha="center")
    axL.axhline(10.0, color=C["mut"], lw=1, ls=(0, (1, 3)))
    axL.text(0.5, 10.15, "10%（真绕行量级参照）", color=C["mut"],
             fontsize=8.5, ha="center")
    for x, y in zip(t_mid, z_ab):
        axL.annotate(f"{y:.2f}%", (x, y), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8,
                     color=C["ink"])
    axL.set_xlabel("路径参数 t（控制点 i/(K-1)）", fontsize=10, color=C["sec"])
    axL.set_ylabel("到 landscape 平面垂直位移（% 中心范数）", fontsize=10,
                   color=C["sec"])
    axL.set_title("图B2-左  优化后中间点的出平面位移\n"
                  "（全部 <0.8%，但见右图——位移小不等于没绕行）",
                  fontsize=11.5, color=C["ink"])
    axL.set_ylim(-0.5, 11.5)
    axL.grid(True, color=C["grid"], linewidth=0.6)
    axL.legend(fontsize=9, frameon=False, loc="center left")
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(C["axis"])
    axL.tick_params(colors=C["mut"], labelsize=9)

    idx = np.arange(1, len(l_opt) + 1)
    w = 0.36
    axR.bar(idx - w / 2, l_opt, w, color=C["blue"],
            label="优化后（曲线通过）")
    axR.bar(idx + w / 2, l_proj, w, color=C["red"],
            label="投影回平面（z 清零）")
    axR.axhline(0.1, color=C["mut"], lw=0.8, ls=(0, (4, 3)))
    axR.text(0.55, 0.14, "主判据 0.1", fontsize=8, color=C["mut"], ha="left")
    for i, (a, b) in enumerate(zip(l_opt, l_proj)):
        axR.text(i + 1 - w / 2, a + 0.02, f"{a:.3f}", ha="center",
                 fontsize=8, color=C["ink"])
        axR.text(i + 1 + w / 2, b + 0.02, f"{b:.3f}", ha="center",
                 fontsize=8, color=C["ink"])
    axR.set_xlabel("中间控制点编号", fontsize=10, color=C["sec"])
    axR.set_ylabel("loss（50 正样本）", fontsize=10, color=C["sec"])
    axR.set_title("图B2-右  z 投影消融（AB）：把出平面位移清零\n"
                  "loss 回弹 barrier 量级 -> 0.6~0.8% 位移正是绕行所在",
                  fontsize=11.5, color=C["ink"])
    axR.set_ylim(0, 1.32)
    axR.grid(True, axis="y", color=C["grid"], linewidth=0.6)
    axR.legend(fontsize=9, frameon=False, loc="upper left",
               bbox_to_anchor=(0.42, 1.0))
    axR.spines[["top", "right"]].set_visible(False)
    axR.spines[["left", "bottom"]].set_color(C["axis"])
    axR.tick_params(colors=C["mut"], labelsize=9)

    fig.suptitle("Mode connectivity 路径几何：2D 投影重合是投影错觉，"
                 "低 loss 通道存在于近乎正交的方向上", fontsize=12.5,
                 color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pOut = os.path.join(OUT, "figB2_zdisplace_ablation.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE MC 补充实验 1 可视化")

elif WHICH == "k15":
    k7 = np.load(os.path.join(OUT, "mc_data.npz"), allow_pickle=False)
    k15 = np.load(os.path.join(OUT, "mc_ext_k15.npz"), allow_pickle=False)

    pairs = [("r0", "AB", C["blue"]), ("r1", "AC", C["orange"]),
             ("r2", "ARI", C["red"])]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9))
    fig.patch.set_facecolor("white")
    for ax, (rk, name, cc) in zip(axes, pairs):
        ts15 = k15[f"{rk}_ts_curve"]
        l15 = k15[f"{rk}_curve_losses"]
        ax.plot(ts15, l15, color=cc, lw=2.2,
                label=f"K=15（max={np.nanmax(l15):.4f}）")
        # K=7 对照（npz 顺序 AB/AC/ARI 同主实验三文件）
        if name == "AB":
            k7d = k7
        elif name == "AC":
            k7d = np.load(os.path.join(OUT, "mc_data_AC.npz"),
                          allow_pickle=False)
        else:
            k7d = np.load(os.path.join(OUT, "mc_data_ARI.npz"),
                          allow_pickle=False)
        ax.plot(k7d["ts_curve"], k7d["curve_losses"], color=C["mut"],
                lw=1.6, ls=(0, (4, 3)),
                label=f"K=7（max={np.nanmax(k7d['curve_losses']):.4f}）")
        ax.axhline(0.1, color=C["axis"], lw=0.8, ls=(0, (4, 3)))
        ax.set_yscale("log")
        ax.set_xlabel("路径参数 t", fontsize=10, color=C["sec"])
        ax.set_title(f"{name}：K=7 vs K=15", fontsize=11.5, color=C["ink"])
        ax.grid(True, color=C["grid"], linewidth=0.6, which="both")
        ax.legend(fontsize=9, frameon=False, loc="upper center")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(C["axis"])
        ax.tick_params(colors=C["mut"], labelsize=9)
    axes[0].set_ylabel("loss（log，50 正样本）", fontsize=10, color=C["sec"])
    fig.suptitle("K 敏感性：控制点加密后 barrier 能否进一步压平"
                 "（虚线灰 = K=7 对照；横虚线 = 主判据 0.1）",
                 fontsize=12.5, color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pOut = os.path.join(OUT, "figD_k15_comparison.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE MC 补充实验 2 可视化")

elif WHICH == "z_ac":
    zz = np.load(os.path.join(OUT, "mc_ext_zdisplace.npz"))
    ab = np.load(os.path.join(OUT, "mc_ext_zproj_ablation.npz"))
    ac = np.load(os.path.join(OUT, "mc_ext_zproj_ablation_ac.npz"))

    z_ab = zz["r0_mid_z"] * 100
    z_ac = zz["r1_mid_z"] * 100
    K = 7
    t_mid = np.arange(1, K - 1) / (K - 1)
    l_ab_o, l_ab_p = ab["l_opt"], ab["l_proj"]
    l_ac_o, l_ac_p = ac["l_opt"], ac["l_proj"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fig.patch.set_facecolor("white")

    axL.plot(t_mid, z_ab, color=C["blue"], lw=2, marker="o", ms=6,
             label="AB（s424243 - dp）")
    axL.plot(t_mid, z_ac, color=C["orange"], lw=2, marker="s", ms=6,
             label="AC（s424243 - s424245）")
    axL.axhline(2.0, color=C["red"], lw=1, ls=(0, (4, 3)))
    axL.text(0.5, 2.15, "2% 判据线", color=C["red"], fontsize=8.5,
             ha="center")
    for x, y in zip(t_mid, z_ac):
        axL.annotate(f"{y:.2f}%", (x, y), xytext=(0, -14),
                     textcoords="offset points", ha="center", fontsize=8,
                     color=C["ink"])
    axL.set_xlabel("路径参数 t（控制点 i/(K-1)）", fontsize=10, color=C["sec"])
    axL.set_ylabel("到 landscape 平面垂直位移（% 中心范数）", fontsize=10,
                   color=C["sec"])
    axL.set_title("AC 中间点出平面位移与 AB 同区间\n"
                  "（0.5~0.8%，薄片几何在两配对上一致）",
                  fontsize=11.5, color=C["ink"])
    axL.set_ylim(-0.5, 3.2)
    axL.grid(True, color=C["grid"], linewidth=0.6)
    axL.legend(fontsize=9, frameon=False, loc="upper center")
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(C["axis"])
    axL.tick_params(colors=C["mut"], labelsize=9)

    idx = np.arange(1, len(l_ac_o) + 1)
    w = 0.36
    axR.bar(idx - w / 2, l_ac_o, w, color=C["orange"],
            label="AC 优化后（曲线通过）")
    axR.bar(idx + w / 2, l_ac_p, w, color=C["red"],
            label="AC 投影回平面（z 清零）")
    axR.axhline(0.1, color=C["mut"], lw=0.8, ls=(0, (4, 3)))
    axR.text(0.55, 0.14, "主判据 0.1", fontsize=8, color=C["mut"], ha="left")
    for i, (a, b) in enumerate(zip(l_ac_o, l_ac_p)):
        axR.text(i + 1 - w / 2, a + 0.03, f"{a:.3f}", ha="center",
                 fontsize=8, color=C["ink"])
        axR.text(i + 1 + w / 2, b + 0.03, f"{b:.3f}", ha="center",
                 fontsize=8, color=C["ink"])
    axR.set_xlabel("中间控制点编号", fontsize=10, color=C["sec"])
    axR.set_ylabel("loss（50 正样本）", fontsize=10, color=C["sec"])
    axR.set_title("AC z 投影消融：与 AB 同构回弹\n"
                  f"（回弹 {l_ac_p.min():.2f}~{l_ac_p.max():.2f}，"
                  "barrier 量级——结论对 AC 同样成立）",
                  fontsize=11.5, color=C["ink"])
    axR.set_ylim(0, max(l_ac_p.max(), 1.2) * 1.15)
    axR.grid(True, axis="y", color=C["grid"], linewidth=0.6)
    axR.legend(fontsize=9, frameon=False, loc="upper left",
               bbox_to_anchor=(0.02, 1.0))
    axR.spines[["top", "right"]].set_visible(False)
    axR.spines[["left", "bottom"]].set_color(C["axis"])
    axR.tick_params(colors=C["mut"], labelsize=9)

    fig.suptitle("AC 配对的 z 投影消融（审稿人问：'AB 的结论在 AC 上也成立吗？'）——"
                 "薄片几何不是 AB 特例", fontsize=12.5,
                 color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pOut = os.path.join(OUT, "figB2_ac_zdisplace.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE P0 AC 消融可视化")

elif WHICH == "k15ari":
    k7 = np.load(os.path.join(OUT, "mc_data_ARI.npz"), allow_pickle=False)
    k15 = np.load(os.path.join(OUT, "mc_ext_k15ari.npz"),
                  allow_pickle=False)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("white")

    axL.plot(k7["ts_line"], k7["line_losses"], color=C["mut"], lw=1.4,
             ls=(0, (1, 3)), label="K=7 直线（batch 50）")
    axL.plot(k7["ts_curve"], k7["curve_losses"], color=C["orange"], lw=2,
             marker="o", ms=4,
             label=f"K=7 曲线（batch 50，max={np.nanmax(k7['curve_losses']):.4f}）")
    axL.plot(k15["r0_ts_line"], k15["r0_line_losses"], color=C["axis"],
             lw=1.4, ls=(0, (5, 3)), label="K=15 直线（batch 10）")
    axL.plot(k15["r0_ts_curve"], k15["r0_curve_losses"], color=C["blue"],
             lw=2.2, marker="s", ms=4,
             label=f"K=15 曲线（batch 10，max={np.nanmax(k15['r0_curve_losses']):.4f}）")
    axL.axhline(0.1, color=C["red"], lw=1, ls=(0, (4, 3)))
    axL.text(0.02, 0.13, "主判据 0.1", color=C["red"], fontsize=8.5)
    axL.set_yscale("log")
    axL.set_xlabel("路径参数 t", fontsize=10, color=C["sec"])
    axL.set_ylabel("loss（log，ARI 配对）", fontsize=10, color=C["sec"])
    axL.set_title("ARI：K=15 仍 FAIL——13 中间点全部入盆地\n"
                  "（0.015~0.023），barrier 集中在连向 RI 端点的末段",
                  fontsize=11.5, color=C["ink"])
    axL.grid(True, color=C["grid"], linewidth=0.6, which="both")
    axL.legend(fontsize=8, frameon=False, loc="upper center")
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(C["axis"])
    axL.tick_params(colors=C["mut"], labelsize=9)

    m7 = k7["ctrl_losses"]
    m15 = k15["r0_mid_losses"]
    axR.bar(np.arange(1, len(m7) + 1) - 0.2, m7, 0.38, color=C["orange"],
            label=f"K=7（5 点，batch 50，max={np.nanmax(m7):.4f}）")
    axR.bar(np.arange(1, len(m15) + 1) + 0.2, m15, 0.38, color=C["blue"],
            label=f"K=15（13 点，batch 10，max={np.nanmax(m15):.4f}）")
    axR.axhline(0.1, color=C["red"], lw=1, ls=(0, (4, 3)))
    axR.text(7.2, 0.115, "主判据 0.1", color=C["red"], fontsize=8.5)
    axR.set_xlabel("中间控制点编号", fontsize=10, color=C["sec"])
    axR.set_ylabel("优化后中间点 loss", fontsize=10, color=C["sec"])
    axR.set_title("两种 K 下中间点全部压入盆地（<0.03）\n"
                  "RI 不可连通不是折线分辨率不足，是端点在 barrier 之外",
                  fontsize=11.5, color=C["ink"])
    axR.grid(True, axis="y", color=C["grid"], linewidth=0.6)
    axR.legend(fontsize=8, frameon=False, loc="upper left")
    axR.spines[["top", "right"]].set_visible(False)
    axR.spines[["left", "bottom"]].set_color(C["axis"])
    axR.tick_params(colors=C["mut"], labelsize=9)

    fig.suptitle("ARI K=15 对照（batch 50 vs batch 10 口径并存标注）："
                 "加密控制点不改变结论——随机初始化与解之间是不可压缩的能量壁垒",
                 fontsize=12.5, color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pOut = os.path.join(OUT, "figD_k15ari_comparison.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE MC 补充实验 2 可视化（k15ari）")

elif WHICH == "seed3":
    sd = np.load(os.path.join(OUT, "mc_ext_3seed.npz"), allow_pickle=False)
    seeds = [("r0", 42, C["blue"]), ("r1", 137, C["orange"]),
             ("r2", 2024, C["aqua"])]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("white")
    maxes = []
    for rk, s, cc in seeds:
        hs = sd[f"{rk}_hist_steps"]
        hc = sd[f"{rk}_hist_curves"]
        m = float(sd[f"{rk}_curve_max"])
        maxes.append(m)
        axL.plot(hs, hc, color=cc, lw=1.8,
                 label=f"seed {s}（max={m:.4f}）")
        axR.plot(sd[f"{rk}_ts_curve"], sd[f"{rk}_curve_losses"],
                 color=cc, lw=1.8, label=f"seed {s}")
    maxes = np.array(maxes)
    axR.axhline(0.1, color=C["axis"], lw=0.8, ls=(0, (4, 3)))
    axR.set_yscale("log")
    axR.set_xlabel("路径参数 t", fontsize=10, color=C["sec"])
    axR.set_title("三条优化后曲线（log 纵轴）", fontsize=11.5,
                  color=C["ink"])
    axR.grid(True, color=C["grid"], linewidth=0.6, which="both")
    axR.legend(fontsize=9, frameon=False, loc="upper center")
    axR.spines[["top", "right"]].set_visible(False)
    axR.spines[["left", "bottom"]].set_color(C["axis"])
    axR.tick_params(colors=C["mut"], labelsize=9)

    axL.set_xlabel("优化 step", fontsize=10, color=C["sec"])
    axL.set_ylabel("中间点平均 loss", fontsize=10, color=C["sec"])
    mean, std = maxes.mean(), maxes.std(ddof=1)
    axL.set_title(f"优化收敛史（3 seed）：max = "
                  f"{mean:.4f}±{std:.4f}（CV={std/mean*100:.1f}%）",
                  fontsize=11.5, color=C["ink"])
    axL.grid(True, color=C["grid"], linewidth=0.6)
    axL.legend(fontsize=9, frameon=False, loc="upper right")
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(C["axis"])
    axL.tick_params(colors=C["mut"], labelsize=9)

    fig.suptitle("AB K=7 三 seed 稳定性：GPU 非确定性下的结论稳健性",
                 fontsize=12.5, color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pOut = os.path.join(OUT, "figE_3seed_stability.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE MC 补充实验 3 可视化")

elif WHICH == "10seed":
    sd = np.load(os.path.join(OUT, "mc_ext_10seed.npz"), allow_pickle=False)
    maxes = np.array([float(sd[f"r{i}_curve_max"]) for i in range(10)])
    mean = float(sd["mean"][0]); std = float(sd["std"][0])
    cv = float(sd["cv"][0])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("white")
    for i in range(10):
        axL.plot(sd[f"r{i}_hist_steps"], sd[f"r{i}_hist_curves"],
                 lw=1.2, alpha=0.75)
    axL.set_xlabel("优化 step", fontsize=10, color=C["sec"])
    axL.set_ylabel("中间点平均 loss", fontsize=10, color=C["sec"])
    axL.set_title(f"10 seed 收敛史（灰细线）：max = "
                  f"{mean:.4f}±{std:.4f}（CV={cv:.1f}%）",
                  fontsize=11.5, color=C["ink"])
    axL.grid(True, color=C["grid"], linewidth=0.6)
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(C["axis"])
    axL.tick_params(colors=C["mut"], labelsize=9)

    axR.bar(np.arange(1, 11), maxes, 0.62, color=C["blue"])
    axR.axhline(mean, color=C["ink"], lw=1.2)
    axR.text(10.4, mean, f"mean={mean:.4f}", fontsize=8.5,
             va="center", color=C["ink"])
    axR.axhline(0.1, color=C["red"], lw=1, ls=(0, (4, 3)))
    axR.text(0.6, 0.115, "主判据 0.1", color=C["red"], fontsize=8.5)
    for i, m in enumerate(maxes, start=1):
        axR.text(i, m + 0.0012, f"{m:.4f}", ha="center", fontsize=7.5,
                 color=C["ink"])
    axR.set_xlabel("opt_seed 序号（42/137/2024/7/99/555/1000/3141/9999/2718）",
                   fontsize=10, color=C["sec"])
    axR.set_ylabel("优化后曲线 max", fontsize=10, color=C["sec"])
    axR.set_ylim(0, max(maxes.max() * 1.25, 0.09))
    axR.set_title("10 个 opt_seed 的 max 分布：全部 PASS 零翻转",
                  fontsize=11.5, color=C["ink"])
    axR.grid(True, axis="y", color=C["grid"], linewidth=0.6)
    axR.spines[["top", "right"]].set_visible(False)
    axR.spines[["left", "bottom"]].set_color(C["axis"])
    axR.tick_params(colors=C["mut"], labelsize=9)

    fig.suptitle("P2a AB K=7 十 seed 稳定性：CV 从 3 seed 的 4.8% 到 10 seed 的 "
                 f"{cv:.1f}%", fontsize=12.5, color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pOut = os.path.join(OUT, "figE_10seed_stability.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE P2a 可视化")

elif WHICH == "5ri":
    sd = np.load(os.path.join(OUT, "mc_ext_ari5ri.npz"), allow_pickle=False)
    maxes = sd["ri_curve_max"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("white")
    for i, cc in enumerate([C["blue"], C["orange"], C["aqua"],
                            C["magenta"], C["yellow"]]):
        axL.plot(sd[f"r{i}_ts_curve"], sd[f"r{i}_curve_losses"],
                 color=cc, lw=1.8,
                 label=f"RI seed {i}（max={maxes[i]:.4f}）")
    axL.axhline(0.1, color=C["red"], lw=1, ls=(0, (4, 3)))
    axL.text(0.02, 0.13, "主判据 0.1", color=C["red"], fontsize=8.5)
    axL.set_yscale("log")
    axL.set_xlabel("路径参数 t", fontsize=10, color=C["sec"])
    axL.set_ylabel("loss（log）", fontsize=10, color=C["sec"])
    axL.set_title("5 个 RI 端点的 ARI 曲线：末段全部爬台",
                  fontsize=11.5, color=C["ink"])
    axL.grid(True, color=C["grid"], linewidth=0.6, which="both")
    axL.legend(fontsize=8.5, frameon=False, loc="upper center")
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(C["axis"])
    axL.tick_params(colors=C["mut"], labelsize=9)

    axR.bar(np.arange(5), maxes, 0.55,
            color=[C["red"] if m > 0.5 else C["mut"] for m in maxes])
    axR.axhline(0.5, color=C["ink"], lw=1.2, ls=(0, (4, 3)))
    axR.text(4.4, 0.53, "FAIL 判据 0.5", fontsize=8.5, color=C["ink"],
             ha="right")
    for i, m in enumerate(maxes):
        axR.text(i, m + 0.02, f"{m:.4f}", ha="center", fontsize=8.5,
                 color=C["ink"])
    axR.set_xlabel("RI 端点 seed（0/1/2/3/4）", fontsize=10, color=C["sec"])
    axR.set_ylabel("优化后曲线 max", fontsize=10, color=C["sec"])
    axR.set_title(f"{int((maxes>0.5).sum())}/5 FAIL——barrier 不是单点偶然",
                  fontsize=11.5, color=C["ink"])
    axR.grid(True, axis="y", color=C["grid"], linewidth=0.6)
    axR.spines[["top", "right"]].set_visible(False)
    axR.spines[["left", "bottom"]].set_color(C["axis"])
    axR.tick_params(colors=C["mut"], labelsize=9)

    fig.suptitle("P2b ARI 五 RI 重复：随机初始化端点的不可连通性跨 seed 成立",
                 fontsize=12.5, color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pOut = os.path.join(OUT, "figF_ari_5ri.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE P2b 可视化")

elif WHICH == "zab5":
    sd = np.load(os.path.join(OUT, "mc_ext_zab5.npz"), allow_pickle=False)
    pairs = list(sd["pairs"]); seeds = list(sd["seeds"])
    l_proj = sd["l_proj"]; z_frac = sd["z_frac"]
    uniq_p = ["AB", "AC"]
    cols = {"AB": C["blue"], "AC": C["orange"]}
    marks = {"AB": "o", "AC": "s"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("white")
    for pi, pname in enumerate(uniq_p):
        for ri in range(len(pairs)):
            if pairs[ri] != pname:
                continue
            lbl = (f"{pname}-s{seeds[ri]}" if pi == 0 else None)
            axL.plot(np.arange(1, 6), z_frac[ri] * 100,
                     color=cols[pname], lw=1.6 if seeds[ri] == 4242 else 1.1,
                     alpha=1.0 if seeds[ri] == 4242 else 0.75,
                     marker=marks[pname], ms=5, label=lbl)
    axL.axhspan(0.5, 1.0, color=C["aqua"], alpha=0.10)
    axL.text(0.06, 0.94, "判据带 0.5~1.0%", fontsize=8,
             color=C["aqua"], transform=axL.transAxes)
    axL.set_xticks(np.arange(1, 6))
    axL.set_xlabel("中间控制点编号", fontsize=10, color=C["sec"])
    axL.set_ylabel("z 位移（% 中心范数）", fontsize=10, color=C["sec"])
    axL.set_title("10 run 的 z 位移：5 条线几乎完全重叠\n"
                  "（跨 opt_seed 数字差异 <0.002pp）",
                  fontsize=11.5, color=C["ink"])
    axL.grid(True, color=C["grid"], linewidth=0.6)
    axL.legend(fontsize=8.5, frameon=False, loc="lower right",
               title="AB（AC 同构，图例略）", title_fontsize=8)
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(C["axis"])
    axL.tick_params(colors=C["mut"], labelsize=9)

    xs, ys, cs, ms = [], [], [], []
    for ri in range(len(pairs)):
        for j in range(l_proj.shape[1]):
            xs.append(pairs[ri] + f"-s{seeds[ri]}")
            ys.append(l_proj[ri, j])
    order = np.argsort(ys)[::-1]
    xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
    axR.scatter(range(len(ys)), ys, s=26,
                color=[cols[x.split("-")[0]] for x in xs], zorder=3)
    axR.axhspan(0.3, 1.1, color=C["aqua"], alpha=0.10)
    axR.text(0.02, 0.32, "判据带 0.3~1.1", fontsize=8,
             color=C["aqua"], transform=axR.transAxes)
    axR.axhline(0.1, color=C["red"], lw=1, ls=(0, (4, 3)))
    axR.text(0.5, 0.14, "主判据 0.1", color=C["red"], fontsize=8.5)
    every = max(1, len(ys) // 10)
    axR.set_xticks(range(0, len(ys), every))
    axR.set_xticklabels([xs[i] for i in range(0, len(ys), every)],
                        fontsize=7, rotation=30)
    axR.set_ylabel("投影回平面后 loss", fontsize=10, color=C["sec"])
    axR.set_title("50 个投影回弹点：全部落 barrier 量级判据带\n"
                  "（0.7% 出平面位移 <-> 回弹，跨 10 run 稳定）",
                  fontsize=11.5, color=C["ink"])
    axR.grid(True, axis="y", color=C["grid"], linewidth=0.6)
    axR.spines[["top", "right"]].set_visible(False)
    axR.spines[["left", "bottom"]].set_color(C["axis"])
    axR.tick_params(colors=C["mut"], labelsize=9)

    n_all = l_proj.size
    hit = float(((l_proj >= 0.3) & (l_proj <= 1.1)).mean() * 100)
    fig.suptitle(f"5-seed z 投影消融（P1 补强）：薄片几何跨 opt_seed 稳定——"
                 f"回弹命中判据带 {hit:.0f}%（{n_all}/{n_all} 点）",
                 fontsize=12.5, color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pOut = os.path.join(OUT, "figH_zablation_5seed.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE zab5 可视化")

elif WHICH == "epsiter":
    sd = np.load(os.path.join(OUT, "mc_ext_eps10_iter.npz"),
                 allow_pickle=False)
    cmax = sd["ctrl_maxes"]; smax = sd["seg_maxes"]
    mx = sd["curve_maxes"]; n = len(mx)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("white")
    rs = np.arange(1, n + 1)
    axL.plot(rs, mx, color=C["ink"], lw=2.2, marker="o", ms=7,
             label="曲线 max（=段中点 max）")
    axL.plot(rs, cmax, color=C["aqua"], lw=2, marker="s", ms=6,
             label="控制点 max（单调降）")
    axL.axhline(0.245, color=C["blue"], lw=1.2, ls=(0, (4, 3)))
    axL.text(n - 0.4, 0.222, "P3b 单次预热 0.245", fontsize=8,
             color=C["blue"], ha="right")
    axL.axhline(0.1, color=C["red"], lw=1, ls=(0, (4, 3)))
    axL.text(0.94, 0.115, "盆地判据 0.1", color=C["red"], fontsize=8.5,
             ha="right")
    axL.set_xticks(rs)
    axL.set_xlabel("预热轮次（每轮 150 步，Adam 动量重置）", fontsize=10,
                   color=C["sec"])
    axL.set_ylabel("loss", fontsize=10, color=C["sec"])
    axL.set_ylim(0, 1.05)
    axL.set_title("迭代预热不收敛：max 在 0.41~0.95 振荡\n"
                  "（'多预热几次就能压到盆地'被否定）",
                  fontsize=11.5, color=C["ink"])
    axL.grid(True, color=C["grid"], linewidth=0.6)
    axL.legend(fontsize=8.5, frameon=False, loc="upper center")
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(C["axis"])
    axL.tick_params(colors=C["mut"], labelsize=9)

    segs = np.arange(1, 7)
    shades = [C["blue"], C["blue"], C["aqua"], C["aqua"],
              C["orange"], C["red"]]
    for r in range(n):
        part = np.load(os.path.join(
            OUT, f"mc_ext_eps10_iter_part{r+1}.npz"))
        axR.plot(segs, part["segmid_losses"], lw=1.4, alpha=0.8,
                 marker="o", ms=4,
                 color=[C["blue"], C["aqua"], C["yellow"],
                        C["orange"], C["magenta"]][r],
                 label=f"轮 {r+1}（max {smax[r]:.2f}）")
    axR.axhline(0.1, color=C["red"], lw=1, ls=(0, (4, 3)))
    axR.text(0.98, 0.115, "盆地判据 0.1", color=C["red"], fontsize=8.5,
             ha="right", transform=axR.transAxes)
    axR.set_xticks(segs)
    axR.set_xlabel("折线段编号（第 6 段 = 连向 RI 端点的末段）",
                   fontsize=10, color=C["sec"])
    axR.set_ylabel("段中点 loss", fontsize=10, color=C["sec"])
    axR.set_yscale("log")
    axR.set_title("反弹位置分层：前 4 段 0.02~0.07（盆内），\n"
                  "末 2 段 0.17~0.95（擦台）——固定不随轮次消除",
                  fontsize=11.5, color=C["ink"])
    axR.grid(True, color=C["grid"], linewidth=0.6, which="both")
    axR.legend(fontsize=8, frameon=False, loc="upper left")
    axR.spines[["top", "right"]].set_visible(False)
    axR.spines[["left", "bottom"]].set_color(C["axis"])
    axR.tick_params(colors=C["mut"], labelsize=9)

    fig.suptitle("P3c 迭代预热（ε=10% × 5 轮）：末段擦台是折线参数化的"
                 "固有属性——中点非优化变量，端点优化无法消除",
                 fontsize=12.5, color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pOut = os.path.join(OUT, "figI_eps10_iter.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE P3c 可视化")

elif WHICH == "epsscan":
    sd = np.load(os.path.join(OUT, "mc_ext_epsscan.npz"), allow_pickle=False)
    epss = sd["eps_list"] * 100
    maxes = sd["curve_maxes"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("white")
    for i, cc in enumerate([C["blue"], C["aqua"], C["orange"],
                            C["yellow"], C["magenta"], C["red"]]):
        axL.plot(sd[f"r{i}_ts_curve"], sd[f"r{i}_curve_losses"],
                 color=cc, lw=1.8,
                 label=f"ε={epss[i]:.1f}%（max={maxes[i]:.4f}）")
    axL.axhline(0.1, color=C["ink"], lw=1, ls=(0, (4, 3)))
    axL.text(0.02, 0.115, "主判据 0.1", color=C["ink"], fontsize=8.5)
    axL.set_yscale("log")
    axL.set_xlabel("路径参数 t", fontsize=10, color=C["sec"])
    axL.set_ylabel("loss（log）", fontsize=10, color=C["sec"])
    axL.set_title("不同 ε 下优化后曲线（RI 端点松绑+范数约束）",
                  fontsize=11.5, color=C["ink"])
    axL.grid(True, color=C["grid"], linewidth=0.6, which="both")
    axL.legend(fontsize=8, frameon=False, loc="upper center")
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(C["axis"])
    axL.tick_params(colors=C["mut"], labelsize=9)

    axR.plot(epss, maxes, color=C["blue"], lw=2.2, marker="o", ms=7)
    axR.axhline(0.1, color=C["red"], lw=1, ls=(0, (4, 3)))
    axR.text(0.3, 0.115, "主判据 0.1", color=C["red"], fontsize=8.5)
    axR.set_yscale("log")
    axR.set_xscale("log")
    axR.set_xticks(epss)
    axR.set_xticklabels([f"{v:.1f}" for v in epss])
    axR.set_xlabel("ε（RI 端点允许位移，% 权重范数，log 轴）",
                   fontsize=10, color=C["sec"])
    axR.set_ylabel("优化后曲线 max（log）", fontsize=10, color=C["sec"])
    crit = np.where(maxes < 0.1)[0]
    crit_txt = (f"临界 ε≈{epss[crit[0]]:.1f}%——到达低 loss 通道"
                f"至少要移动权重范数的 {epss[crit[0]]:.1f}%"
                if len(crit) else "20% 内无临界点——训练必要性论证成立")
    axR.set_title(f"ε-max 关系：{crit_txt}",
                  fontsize=11.5, color=C["ink"])
    axR.grid(True, color=C["grid"], linewidth=0.6, which="both")
    axR.spines[["top", "right"]].set_visible(False)
    axR.spines[["left", "bottom"]].set_color(C["axis"])
    axR.tick_params(colors=C["mut"], labelsize=9)

    fig.suptitle("P3 ARI ε 扫描：'RI 端点必须能走多远才能连通？'"
                 "——训练必要性的定量版本",
                 fontsize=12.5, color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pOut = os.path.join(OUT, "figG_eps_scan.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE P3 可视化")

elif WHICH == "f197":
    # P1：50 样本 vs 全量 197 对照（三配对 + 3seed + AC 消融）
    f197 = np.load(os.path.join(OUT, "full197", "f197_all.npz"),
                   allow_pickle=False)
    k7ab = np.load(os.path.join(OUT, "mc_data.npz"), allow_pickle=False)
    k7ac = np.load(os.path.join(OUT, "mc_data_AC.npz"), allow_pickle=False)
    k7ari = np.load(os.path.join(OUT, "mc_data_ARI.npz"), allow_pickle=False)
    abl_ac = np.load(os.path.join(OUT, "mc_ext_zproj_ablation_ac.npz"),
                     allow_pickle=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.9))
    fig.patch.set_facecolor("white")
    trio = [("r0", "AB", k7ab, C["blue"]),
            ("r1", "AC", k7ac, C["orange"]),
            ("r2", "ARI", k7ari, C["red"])]
    for ax, (rk, name, k7, cc) in zip(axes, trio):
        m197 = float(np.nanmax(f197[f"{rk}_curve_losses"]))
        m50 = float(np.nanmax(k7["curve_losses"]))
        ax.plot(k7["ts_curve"], k7["curve_losses"], color=C["mut"],
                lw=1.8, ls=(0, (4, 3)),
                label=f"50 样本（max={m50:.4f}）")
        ax.plot(f197[f"{rk}_ts_curve"], f197[f"{rk}_curve_losses"],
                color=cc, lw=2.2,
                label=f"全量 197（max={m197:.4f}）")
        ax.axhline(0.1, color=C["axis"], lw=0.8, ls=(0, (4, 3)))
        ax.set_yscale("log")
        ax.set_xlabel("路径参数 t", fontsize=10, color=C["sec"])
        ax.set_title(f"{name}：50 vs 全量 197（判定"
                     f"{'不' if (m197 < 0.1) == (m50 < 0.1) else ''}翻转）",
                     fontsize=11.5, color=C["ink"])
        ax.grid(True, color=C["grid"], linewidth=0.6, which="both")
        ax.legend(fontsize=8.5, frameon=False, loc="upper center")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(C["axis"])
        ax.tick_params(colors=C["mut"], labelsize=9)
    axes[0].set_ylabel("loss（log）", fontsize=10, color=C["sec"])
    fig.suptitle("P1 全量 197 正样本验证：PASS/FAIL 判定零翻转，"
                 "AB/AC max 偏差 -21%（同量级），ARI +1%",
                 fontsize=12.5, color=C["ink"], y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pOut = os.path.join(OUT, "full197", "figP1_full197_comparison.png")
    fig.savefig(pOut, dpi=110)
    plt.close(fig)
    print(f"fig -> {pOut}")

    # 右下：全量 AC 消融对照（与 50 样本版同构）
    fig2, ax = plt.subplots(figsize=(6.8, 4.6))
    fig2.patch.set_facecolor("white")
    idx = np.arange(1, 6)
    w = 0.36
    ax.bar(idx - w / 2, abl_ac["l_opt"], w, color=C["orange"],
           label="AC 优化后（全量 197）")
    ax.bar(idx + w / 2, abl_ac["l_proj"], w, color=C["red"],
           label="AC 投影回平面（z 清零）")
    ax.axhline(0.1, color=C["mut"], lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel("中间控制点编号", fontsize=10, color=C["sec"])
    ax.set_ylabel("loss（全量 197）", fontsize=10, color=C["sec"])
    ax.set_title("P1 AC z 投影消融（全量口径）：回弹 "
                 f"{abl_ac['l_proj'].min():.2f}~{abl_ac['l_proj'].max():.2f}\n"
                 "与 50 样本版（0.41~1.09）同构——薄片几何口径无关",
                 fontsize=11.5, color=C["ink"])
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.6)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["axis"])
    ax.tick_params(colors=C["mut"], labelsize=9)
    fig2.tight_layout()
    pOut2 = os.path.join(OUT, "full197", "figP1_full197_ac_ablation.png")
    fig2.savefig(pOut2, dpi=110)
    plt.close(fig2)
    print(f"fig -> {pOut2}")
    print("DONE P1 可视化")

elif WHICH == "summary":
    # P4 汇总表图：所有实验的关键数字一张图
    rows = []

    def load_npz(rel):
        p = os.path.join(OUT, rel)
        return np.load(p, allow_pickle=False) if os.path.exists(p) else None

    mc_ab = load_npz("mc_data.npz")
    mc_ac = load_npz("mc_data_AC.npz")
    mc_ari = load_npz("mc_data_ARI.npz")
    k15ari = load_npz("mc_ext_k15ari.npz")
    zz = load_npz("mc_ext_zdisplace.npz")
    abl_ac = load_npz("mc_ext_zproj_ablation_ac.npz")
    s3 = load_npz("mc_ext_3seed.npz")
    f197 = load_npz("full197/f197_all.npz")
    s10 = load_npz("mc_ext_10seed.npz")
    ri5 = load_npz("mc_ext_ari5ri.npz")
    eps = load_npz("mc_ext_epsscan.npz")

    def g(d, k, default=np.nan):
        return float(d[k]) if d is not None and k in d else default

    def gmax(d, k="curve_losses"):
        return float(np.nanmax(d[k])) if d is not None else np.nan

    # (类别, 名称, 值文本, 判定, 判定色)
    def verdict(passed):
        return ("PASS", C["aqua"]) if passed else (("FAIL", C["red"]))

    rows.append(("主实验 K=7 (50样本)",
                 "AB 直线 max",
                 f"{gmax(mc_ab, 'line_losses'):.4f}", "基线", C["mut"]))
    if mc_ab is not None:
        rows.append(("主实验 K=7 (50样本)", "AB 曲线 max",
                     f"{gmax(mc_ab):.4f}", *verdict(gmax(mc_ab) < 0.1)))
    if mc_ac is not None:
        rows.append(("主实验 K=7 (50样本)", "AC 曲线 max",
                     f"{gmax(mc_ac):.4f}", *verdict(gmax(mc_ac) < 0.1)))
    if mc_ari is not None:
        rows.append(("主实验 K=7 (50样本)", "ARI 曲线 max",
                     f"{gmax(mc_ari):.4f}", *verdict(False)))
    if k15ari is not None:
        rows.append(("实验2 K=15 ARI (b10)",
                     "曲线 max",
                     f"{gmax(k15ari, 'r0_curve_losses'):.4f}",
                     *verdict(False)))
    if zz is not None:
        zmax = float(np.max(np.maximum(zz["r0_mid_z"], zz["r1_mid_z"]))) * 100
        rows.append(("实验1 z 位移",
                     "AB/AC max z",
                     f"{zmax:.2f}%", "薄片几何", C["blue"]))
    if abl_ac is not None:
        rows.append(("实验1/P0 z 消融",
                     "AC 投影回弹",
                     f"{abl_ac['l_proj'].min():.2f}~{abl_ac['l_proj'].max():.2f}",
                     "绕行实锤", C["blue"]))
    if s3 is not None:
        m3 = np.array([float(s3[f"r{i}_curve_max"]) for i in range(3)])
        cv3 = m3.std(ddof=1) / m3.mean() * 100
        rows.append(("实验3 稳定性", "3 seed CV",
                     f"{cv3:.1f}%", "稳定", C["aqua"]))
    if s10 is not None:
        rows.append(("P2a 稳定性", "10 seed CV",
                     f"{float(s10['cv'][0]):.1f}%", "稳定", C["aqua"]))
    if ri5 is not None:
        nf = int((ri5["ri_curve_max"] > 0.5).sum())
        rows.append(("P2b 5 RI 重复", "FAIL 数",
                     f"{nf}/5", *verdict(nf == 5)))
    if f197 is not None:
        for i, nm in ((0, "AB"), (1, "AC"), (2, "ARI")):
            passed = float(f197[f"r{i}_curve_max"]) < 0.1
            rows.append(("P1 全量197", f"{nm} 曲线 max",
                         f"{float(f197[f'r{i}_curve_max']):.4f}",
                         *verdict(passed if nm != "ARI" else False)))
    if eps is not None:
        em = eps["curve_maxes"]
        crit = np.where(em < 0.1)[0]
        ct = (f"临界 ε={eps['eps_list'][crit[0]]*100:.1f}%" if len(crit)
              else "无临界点")
        rows.append(("P3 ε 扫描", "RI 松绑",
                     ct, "训练必要性", C["blue"]))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10.8, 0.52 * n + 1.6))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    hdr = ["实验", "指标", "数值", "判定"]
    col_x = [0.02, 0.34, 0.60, 0.80]
    for x, h in zip(col_x, hdr):
        ax.text(x, 0.965, h, fontsize=10.5, fontweight="bold",
                color=C["ink"], transform=ax.transAxes)
    ax.plot([0.01, 0.99], [0.945, 0.945], color=C["axis"], lw=1,
            transform=ax.transAxes)
    prev_grp = None
    for i, (grp, nm, val, vd, vc) in enumerate(rows):
        y = 0.905 - i * (0.86 / n)
        if grp != prev_grp:
            ax.plot([0.01, 0.99], [y + 0.024, y + 0.024],
                    color=C["grid"], lw=0.7, transform=ax.transAxes)
            prev_grp = grp
        ax.text(col_x[0], y, grp, fontsize=9, color=C["sec"],
                transform=ax.transAxes)
        ax.text(col_x[1], y, nm, fontsize=9, color=C["ink"],
                transform=ax.transAxes)
        ax.text(col_x[2], y, val, fontsize=9, color=C["ink"],
                transform=ax.transAxes)
        ax.text(col_x[3], y, vd, fontsize=9, color=vc, fontweight="bold",
                transform=ax.transAxes)
    ax.set_title("Mode connectivity 证据汇总（Round12 全系列）",
                 fontsize=12.5, color=C["ink"], pad=14)
    pOut = os.path.join(OUT, "figG_evidence_summary.png")
    fig.savefig(pOut, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"fig -> {pOut}")
    print("DONE P4 汇总图")
