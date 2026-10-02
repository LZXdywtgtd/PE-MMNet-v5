# -*- coding: utf-8 -*-
"""Round12 工作项 3A：收敛点分布图（零训练）。

主图：landscape 热图底 + 6 seed 收敛点 + 随机初始化远点 + 3 个真实对照点
副图：收敛区放大（6 点分布可读）

口径：
- 6 收敛点 = 6 个独立训练（不同 seed）的 best.pt 终点分布，非单次训练
  轨迹（图注明示）
- 执行期核实：v5a6_dprime_augfix2 / v5a6_dprime_kd2 与 resnet18 键集
  形状完全一致（205 键）→ 可精确投影本平面（修正计划"不能投影"假设）；
  swin_yolo 259 键不同参数空间，仍只作量级参照（loss=0.0239）
- augfix2/kd2 为旧 GT（L 形污染期）训练，在本干净 GT val 上评的 loss
  仅作跨代参照
- dp latest.pt（ep150）vs best.pt（ep138）对照 = V5-030 教训可视化
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

from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import OrderedKeypointLoss

from _viz_common import setup_font
setup_font()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300",
     "violet": "#4a3aa7", "red": "#e34948",
     "ink": "#0b0b0b", "sec": "#52514e", "mut": "#898781",
     "grid": "#e1e0d9", "axis": "#c3c2b7"}
SEED_C = {"v5b_s3_gtfix_dp": C["blue"], "v5b_s3_gtfix_base": C["blue"],
          "v5b_s4_seed424243": C["orange"], "v5b_s4_seed424244": C["aqua"],
          "v5b_s4_seed424245": C["yellow"], "v5b_s4_seed424246": C["magenta"]}
SEED_LB = {"v5b_s3_gtfix_dp": "dp(seed424242)", "v5b_s3_gtfix_base": "gtfix_base",
           "v5b_s4_seed424243": "seed424243", "v5b_s4_seed424244": "seed424244",
           "v5b_s4_seed424245": "seed424245", "v5b_s4_seed424246": "seed424246"}

LAND_OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                        "round12_loss_landscape")
OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_convergence_map")
os.makedirs(OUT, exist_ok=True)

VAL_CACHE = os.path.join(PROJECT_ROOT, "logs", "sim_cache",
                         "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
CHAMP = "v5b_s3_gtfix_dp"
AXIS_RUNS = ["v5b_s4_seed424243", "v5b_s4_seed424245"]
NEW_RUNS = [f"{CHAMP}#latest", "v5a6_dprime_augfix2", "v5a6_dprime_kd2"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEV}")

z = np.load(os.path.join(LAND_OUT, "landscape_data.npz"), allow_pickle=False)
anchor_runs = [str(s) for s in z["anchor_runs"]]
anchor_coord = z["anchor_coord"]
anchor_loss = z["anchor_loss"]
center_loss = float(z["center_loss"][0])
swin_loss = float(z["swin_loss"][0])
sample_idx = z["sample_idx"]
a1s, a2s = z["alphas1"], z["alphas2"]
loss_grid = z["loss_grid"]


def build_model(variant="resnet18"):
    kw = dict(image_channels=3, image_size=128, pretrained_2d=False,
              use_gp=True, min_kpts=8, max_kpts=16)
    if variant == "resnet18":
        kw["spatial_head"] = False
    return create_v5_model(variant, **kw)


def flat_state(model):
    return np.concatenate(
        [v.detach().double().cpu().numpy().ravel()
         for v in model.state_dict().values()
         if v.dtype.is_floating_point])


def load_flat(run):
    if run == "__random_init__":
        m = build_model().to(DEV)
        return flat_state(m)
    name = run.split("#")[0]
    ck_name = "latest.pt" if "#" in run else "best.pt"
    load_checkpoint(os.path.join(TH, name, "checkpoints", ck_name),
                    model, map_location=DEV)
    return flat_state(model)


# ---------- 模型 + 平面基向量（重建以投影新点）----------
model = build_model().to(DEV)
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

theta_c = load_flat(CHAMP)
M = np.stack([load_flat(AXIS_RUNS[0]) - theta_c,
              load_flat(AXIS_RUNS[1]) - theta_c])
Ginv = np.linalg.inv(M @ M.T)

d = np.load(VAL_CACHE)
x1 = torch.from_numpy(d["x_1d"][sample_idx]).float().to(DEV)
x2 = torch.from_numpy(d["x_2d"][sample_idx]).float().to(DEV)
pk = torch.from_numpy(d["true_crack_pixels"][sample_idx]).float().to(DEV)
pm = torch.from_numpy(d["pixel_mask"][sample_idx]).to(DEV)
tk = torch.from_numpy(d["true_keypoints"][sample_idx]).float().to(DEV)
km = torch.from_numpy(d["keypoint_mask"][sample_idx]).to(DEV)
tb = torch.from_numpy(d["true_bbox"][sample_idx]).float().to(DEV)

loss_fn = OrderedKeypointLoss(
    lambda_coverage=1.0, lambda_bbox=1.0, lambda_gp=0.1,
    lambda_poisson=0.05, lambda_kpt_direct=0.5,
    lambda_ordered=0.0, lambda_validity=0.0,
    lambda_position_6d=0.0, lambda_uv_projection=0.0)


@torch.no_grad()
def eval_flat(flat):
    try:
        sd = {}
        ptr = 0
        for k, v in base_state.items():
            if v.dtype.is_floating_point:
                n = v.numel()
                sd[k] = torch.from_numpy(
                    flat[ptr:ptr + n].astype(np.float32)).reshape(v.shape)
                ptr += n
            else:
                sd[k] = base_state[k]
        model.load_state_dict(sd)
        model.eval()
        out = model(x1, x2)
        lo = loss_fn(
            pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
            true_bbox=tb, true_crack_pixels=pk,
            gp_module=getattr(model, "gp_module", None),
            pixel_mask=pm, true_kpts=tk, kpt_mask=km,
            pred_validity=out.get("validity"),
            true_position_6d=None,
            pred_position_6d=out.get("position_6d"))
        t = lo["total"].item()
        return t if np.isfinite(t) else np.nan
    except Exception:
        return np.nan


# ---------- 新点：投影 + 评 loss ----------
new_coord, new_loss, new_ep = {}, {}, {}
for run in NEW_RUNS:
    th = load_flat(run)
    a = (th - theta_c) @ M.T @ Ginv
    tl = eval_flat(th)
    name = run.split("#")[0]
    ck = torch.load(os.path.join(TH, name, "checkpoints",
                                 "latest.pt" if "#" in run else "best.pt"),
                    map_location="cpu", weights_only=False)
    new_coord[run] = a
    new_loss[run] = tl
    new_ep[run] = int(ck["epoch"])
    print(f"new {run:24s} α=({a[0]:+.3f},{a[1]:+.3f})  "
          f"loss={tl:.4f}  ep={new_ep[run]}")

# ---------- 图 ----------
cmap = LinearSegmentedColormap.from_list(
    "loss_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
cmap.set_bad("#e8e7e2")
A1, A2 = np.meshgrid(a1s, a2s, indexing="ij")

fig, (ax, axz) = plt.subplots(1, 2, figsize=(14.5, 7),
                              gridspec_kw={"width_ratios": [1, 1]})
fig.patch.set_facecolor("white")

for a in (ax, axz):
    a.pcolormesh(A1, A2, loss_grid, cmap=cmap,
                 vmin=np.nanmin(loss_grid), vmax=np.nanmax(loss_grid),
                 shading="nearest")
    a.plot([0, 1, 0, 0], [0, 0, 1, 0], color=C["axis"], lw=0.8,
           ls=(0, (4, 3)), zorder=2)
    a.spines[["top", "right"]].set_visible(False)
    a.spines[["left", "bottom"]].set_color(C["axis"])
    a.tick_params(colors=C["mut"], labelsize=9)
ax.set_xlabel("α1（→ seed424243 方向）", fontsize=10, color=C["sec"])
ax.set_ylabel("α2（→ seed424245 方向）", fontsize=10, color=C["sec"])

# 收敛点：dp 中心 (0,0) + npz 里 5 锚点（base/424243/424244/424245/424246）
seed_pts = [(0.0, 0.0, center_loss, CHAMP)]
for run, (a1, a2), lv in zip(anchor_runs, anchor_coord, anchor_loss):
    if run == "__random_init__":
        ri_xy = (a1, a2)
        ri_loss = lv
        continue
    seed_pts.append((a1, a2, lv, run))
for a1, a2, lv, run in seed_pts:
    ax.scatter(a1, a2, s=80, c=SEED_C[run], marker="o", zorder=4,
               edgecolors="white", linewidths=1.4)
ax.scatter(*ri_xy, s=100, c=C["ink"], marker="X", zorder=4,
           edgecolors="white", linewidths=1.4)

# 新点：latest 三角 / augfix2 菱形 / kd2 菱形
NEW_STYLE = {f"{CHAMP}#latest": (C["violet"], "triangle"),
             "v5a6_dprime_augfix2": (C["red"], "diamond"),
             "v5a6_dprime_kd2": (C["green"], "diamond")}
for run, (cc, mk) in NEW_STYLE.items():
    a1, a2 = new_coord[run]
    ax.scatter(a1, a2, s=95, c=cc, marker={"triangle": "^", "diamond": "D"}[mk],
               zorder=4, edgecolors="white", linewidths=1.4)

cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, shrink=0.85)
cb.set_label("loss（50 正样本 val，OrderedKeypointLoss total）",
             fontsize=9, color=C["sec"])
cb.ax.tick_params(colors=C["mut"], labelsize=8)

ax.set_title(
    "图A  收敛点分布：6 个独立训练（不同 seed）的 best.pt 终点\n"
    "（插值平面同工作项 2；这是终点分布，非单次训练轨迹）",
    fontsize=11.5, color=C["ink"])

# 副图：收敛区放大
zs = [(a1, a2, lv, run) for a1, a2, lv, run in seed_pts]
zs += [(new_coord[r][0], new_coord[r][1], new_loss[r], r) for r in NEW_RUNS]
xs = [p[0] for p in zs]
ys = [p[1] for p in zs]
pad = 0.08
axz.set_xlim(min(xs) - pad, max(xs) + pad)
axz.set_ylim(min(ys) - pad, max(ys) + pad)
axz.set_xlabel("α1（放大）", fontsize=10, color=C["sec"])
axz.set_ylabel("α2（放大）", fontsize=10, color=C["sec"])
axz.set_title("图B  收敛区放大：6 seed 终点聚集于同一盆地\n"
              "（加 latest/augfix2/kd2 对照点）",
              fontsize=11.5, color=C["ink"])

# 放大面板标注：逐点 (offset, ha) 防重叠（dp/latest 近、base/augfix2 近、
# 424244/424246 近，各给相反方向）
OFF_Z = {CHAMP: ((12, 12), "left"), f"{CHAMP}#latest": ((-14, 26), "right"),
         "v5b_s3_gtfix_base": ((12, -30), "left"),
         "v5a6_dprime_augfix2": ((12, 8), "left"),
         "v5b_s4_seed424243": ((-12, -36), "right"),
         "v5b_s4_seed424245": ((12, -8), "left"),
         "v5b_s4_seed424244": ((12, 8), "left"),
         "v5b_s4_seed424246": ((12, -4), "left"),
         "v5a6_dprime_kd2": ((12, -6), "left"),
         "__random_init__": ((12, 12), "left")}
ALL_PTS = [(r, (a1, a2), lv) for a1, a2, lv, r in seed_pts]
ALL_PTS += [(r, (new_coord[r][0], new_coord[r][1]), new_loss[r])
            for r in NEW_RUNS]
for run, (a1, a2), lv in ALL_PTS:
    if run == CHAMP:
        cc, mk, tag = SEED_C[run], "o", f"{SEED_LB[run]} (best ep138)"
    elif run in SEED_LB:
        cc, mk, tag = SEED_C[run], "o", f"{SEED_LB[run]}"
    else:
        cc, mk = NEW_STYLE[run]
        mk = {"triangle": "^", "diamond": "D"}[mk]
        tag = {"v5a6_dprime_augfix2": "augfix2(旧GT)",
               "v5a6_dprime_kd2": "kd2 λ=2.0(旧GT)"}.get(
                   run, f"dp latest ep{new_ep[run]}")
    axz.scatter(a1, a2, s=80, c=cc, marker=mk, zorder=4,
                edgecolors="white", linewidths=1.4)
    (off, ha) = OFF_Z[run]
    axz.annotate(f"{tag}\nloss={lv:.3f}", (a1, a2),
                 xytext=off, textcoords="offset points", fontsize=8,
                 color=C["ink"], zorder=5, ha=ha,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                           alpha=0.78))
ri_tag = f"随机初始化\nloss={ri_loss:.3f}"
(off, ha) = OFF_Z["__random_init__"]
axz.annotate(ri_tag, ri_xy, xytext=off, textcoords="offset points",
             fontsize=8, color=C["ink"], zorder=5, ha=ha,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none",
                       alpha=0.78))

fig.suptitle(
    f"Round12-3A  收敛点分布图（零训练；swin_yolo 不同参数空间无法投影，"
    f"量级参照 loss={swin_loss:.4f}）",
    fontsize=12.5, color=C["ink"])
fig.text(0.01, 0.015,
         "注：augfix2/kd2 为旧 GT（L 形污染期）训练，在干净 GT val 上评的 loss 仅作跨代参照；"
         "swin_yolo 259 键不同架构，不画假点。平面定义见 landscape_data.npz。",
         fontsize=8, color=C["mut"])
fig.subplots_adjust(bottom=0.12, wspace=0.22)
pOut = os.path.join(OUT, "fig_convergence_map.png")
fig.savefig(pOut, dpi=110)
plt.close(fig)
print(f"fig -> {pOut}")

np.savez_compressed(
    os.path.join(OUT, "convergence_extra.npz"),
    new_runs=np.array(NEW_RUNS, dtype=np.str_),
    new_coord=np.stack([new_coord[r] for r in NEW_RUNS]),
    new_loss=np.array([new_loss[r] for r in NEW_RUNS]),
    new_epoch=np.array([new_ep[r] for r in NEW_RUNS]))
print("DONE round12 工作项3A")
