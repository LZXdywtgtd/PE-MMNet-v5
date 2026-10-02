# -*- coding: utf-8 -*-
"""Round12 工作项 4：Garipov 折线 mode connectivity。

问题：Round12 工作项 2 证实 6 个收敛解与随机初始化之间存在线性插值
barrier（mid-loss≈1.10）。本脚本验证这些解是否位于同一连通低 loss
区域——Garipov et al. 2018 折线模式连接（piecewise linear path）。

方法：
- 端点 A=v5b_s4_seed424243 best、B=v5b_s3_gtfix_dp best（固定）
- K=7 控制点 θ_i，初始在直线上；只优化中间 K-2=5 个
- loss=OrderedKeypointLoss total，50 正样本单 batch（同工作项 2 口径）
- Adam lr=1e-3，S=150 步；梯度同时回传所有中间点
- 前向用 torch.func.functional_call——控制点是 leaf 张量，梯度穿图

成功判据：优化后曲线 max loss < 0.1（主）；>0.5 停（拍板）。
落盘 mc_data.npz；A↔B 打通则追加 A↔C、A↔随机初始化。
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
from torch.func import functional_call

from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import OrderedKeypointLoss

from _viz_common import setup_font
setup_font()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "green": "#008300", "ink": "#0b0b0b", "sec": "#52514e",
     "mut": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
     "red": "#e34948"}

LAND_OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                        "round12_loss_landscape")
OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_mode_connectivity")
os.makedirs(OUT, exist_ok=True)
VAL_CACHE = os.path.join(PROJECT_ROOT, "logs", "sim_cache",
                         "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 7
STEPS = 150
LR = 1e-3
N_SAMPLES = 50
SEED = 424242
PAIR = sys.argv[1] if len(sys.argv) > 1 else "AB"   # AB | AC | ARI
TAG = "" if PAIR == "AB" else f"_{PAIR}"
print(f"device = {DEV}  pair = {PAIR}")


def build_model():
    return create_v5_model(
        "resnet18", image_channels=3, image_size=128, pretrained_2d=False,
        use_gp=True, min_kpts=8, max_kpts=16, spatial_head=False).to(DEV)


def flat_state(model):
    return np.concatenate(
        [v.detach().double().cpu().numpy().ravel()
         for v in model.state_dict().values()
         if v.dtype.is_floating_point])


def load_sd(run):
    ck = torch.load(os.path.join(TH, run, "checkpoints", "best.pt"),
                    map_location="cpu", weights_only=False)
    return ck.get("model_state_dict", ck)


model = build_model()
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
FLOAT_KEYS = [k for k, v in base_state.items()
              if v.dtype.is_floating_point]
# cudnn batch_norm 对 running stats 不可微——只对可学习参数开梯度，
# buffer（running_mean/var 等）照插值但 detach
PARAM_KEYS = set(dict(model.named_parameters()).keys())

# ---------- 数据（同工作项 2 的 50 样本）----------
z = np.load(os.path.join(LAND_OUT, "landscape_data.npz"),
            allow_pickle=False)
sample_idx = z["sample_idx"]
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


def make_leaf(sd_like):
    return {k: (v.detach().clone().to(DEV).requires_grad_(True)
                if k in PARAM_KEYS else v.detach().clone().to(DEV))
            for k, v in sd_like.items()}


@torch.no_grad()
def eval_sd(sd, params=None):
    if params is not None:
        model.load_state_dict(
            {k: (v.detach() if v.requires_grad else v)
             for k, v in params.items()}, strict=True)
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
    return lo


# ---------- 端点 ----------
def random_init_sd(seed=0):
    torch.manual_seed(seed)
    m = build_model()
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


sd_A = load_sd("v5b_s4_seed424243")
if PAIR == "AB":
    sd_B = load_sd("v5b_s3_gtfix_dp")
    label_B = "B: dp 收官"
elif PAIR == "AC":
    sd_B = load_sd("v5b_s4_seed424245")
    label_B = "C: seed424245"
else:  # ARI
    sd_B = random_init_sd(0)
    label_B = "RI: 随机初始化"
theta_A = make_leaf(sd_A)
theta_B = make_leaf(sd_B)
with torch.no_grad():
    lA = eval_sd(None, theta_A)["total"].item()
    lB = eval_sd(None, theta_B)["total"].item()
print(f"端点 A(s424243)={lA:.4f}  {label_B}={lB:.4f}")

# 控制点：端点别名 + 中间 leaf（直线上初始化）
mids = []
for i in range(1, K - 1):
    t = i / (K - 1)
    mids.append(make_leaf(sd_A))
for p_mid, i in zip(mids, range(1, K - 1)):
    t = i / (K - 1)
    with torch.no_grad():
        for k in FLOAT_KEYS:
            p_mid[k].copy_(theta_A[k].detach() *
                           (1 - t) + theta_B[k].detach() * t)


def path_points():
    return [theta_A] + mids + [theta_B]


def curve_loss():
    pts = path_points()
    model.eval()  # BN eval 模式：train 模式会原地更新 running stats（leaf 禁改）
    total = 0.0
    for th in pts[1:-1]:
        out = functional_call(model, th, (x1, x2))
        lo = loss_fn(
            pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
            true_bbox=tb, true_crack_pixels=pk,
            gp_module=None, pixel_mask=pm, true_kpts=tk, kpt_mask=km,
            pred_validity=out.get("validity"),
            true_position_6d=None,
            pred_position_6d=out.get("position_6d"))
        total = total + lo["total"]
    return total


@torch.no_grad()
def sample_path_losses(tag):
    """沿折线 100 点采样 loss（线性段插值，端点段也在内）。"""
    ts = np.linspace(0.0, 1.0, 100)
    losses = np.full(100, np.nan)
    pts = path_points()
    with torch.no_grad():
        for j, t in enumerate(ts):
            x = t * (K - 1)               # 折线弧参数 ∈ [0, K-1]
            seg = int(np.clip(np.floor(x), 0, K - 2))
            u = x - seg                   # 段内插值系数
            sd_mix = {}
            for k in FLOAT_KEYS:
                sd_mix[k] = (pts[seg][k].detach() * (1 - u) +
                             pts[seg + 1][k].detach() * u)
            for k, v in base_state.items():
                if k not in sd_mix:
                    sd_mix[k] = v
            model.load_state_dict(sd_mix, strict=True)
            model.eval()
            out = model(x1, x2)
            lo = loss_fn(
                pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
                true_bbox=tb, true_crack_pixels=pk,
                gp_module=None, pixel_mask=pm, true_kpts=tk, kpt_mask=km,
                pred_validity=out.get("validity"),
                true_position_6d=None,
                pred_position_6d=out.get("position_6d"))
            v = lo["total"].item()
            losses[j] = v if np.isfinite(v) else np.nan
    print(f"[{tag}] path: min={np.nanmin(losses):.4f} "
          f"max={np.nanmax(losses):.4f} mean={np.nanmean(losses):.4f}")
    return ts, losses


# ---------- 直线基线 ----------
ts, line_losses = sample_path_losses("直线基线")

# ---------- 优化 ----------
mid_params = [th[k] for th in mids for k in PARAM_KEYS]
opt = torch.optim.Adam(mid_params, lr=LR)
hist_every = 5
hist_steps, hist_curves = [], []
print("开始优化……")
for step in range(STEPS):
    opt.zero_grad(set_to_none=True)
    loss = curve_loss()
    if not torch.isfinite(loss):
        print(f"step {step}: loss={loss.item()} NaN/Inf——发散，停")
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
    opt.step()
    if step % hist_every == 0 or step == STEPS - 1:
        hist_steps.append(step)
        hist_curves.append(loss.item() / (K - 2))
        print(f"step {step:3d}  mean_mid_loss={loss.item()/(K-2):.4f}")

# ---------- 优化后曲线 ----------
ts2, curve_losses = sample_path_losses("优化后")

# 优化后中间控制点各自 loss（npz 记录用）
mid_losses = []
with torch.no_grad():
    model.eval()
    for th in mids:
        out = functional_call(model, th, (x1, x2))
        lo = loss_fn(
            pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
            true_bbox=tb, true_crack_pixels=pk,
            gp_module=None, pixel_mask=pm, true_kpts=tk, kpt_mask=km,
            pred_validity=out.get("validity"),
            true_position_6d=None,
            pred_position_6d=out.get("position_6d"))
        mid_losses.append(lo["total"].item())
print("优化后中间点 loss: " +
      " ".join(f"{v:.4f}" for v in mid_losses))

# ---------- 投影坐标（复用工作项 2 平面；与配对无关的正则平面）----------
def flat_np(sd):
    return np.concatenate(
        [sd[k].detach().cpu().numpy().astype(np.float64).ravel()
         for k in FLOAT_KEYS])


theta_c_np = flat_np(load_sd("v5b_s3_gtfix_dp"))
M = np.stack([flat_np(load_sd("v5b_s4_seed424243")) - theta_c_np,
              flat_np(load_sd("v5b_s4_seed424245")) - theta_c_np])
Ginv = np.linalg.inv(M @ M.T)


def project_pts(pts_sd_list):
    out = []
    for sd in pts_sd_list:
        diff = flat_np(sd) - theta_c_np
        out.append(diff @ M.T @ Ginv)
    return np.stack(out)


pts_sd = path_points()
proj_ctrl = project_pts(pts_sd)

# 直线/曲线的 100 采样点也投影（每点构造混合 sd 太重——投影只需
# 中间点坐标，曲线在 α 平面 = 分段线性连接控制点投影）
line_proj = np.linspace(proj_ctrl[0], proj_ctrl[-1], 100)

mc_line_max = np.nanmax(line_losses)
mc_curve_max = np.nanmax(curve_losses)
print(f"直线 max={mc_line_max:.4f}  曲线 max={mc_curve_max:.4f}  "
      f"比值={mc_curve_max/max(mc_line_max,1e-9):.2f}")

np.savez_compressed(
    os.path.join(OUT, f"mc_data{TAG}.npz"),
    ts_line=ts, line_losses=line_losses,
    ts_curve=ts2, curve_losses=curve_losses,
    hist_steps=np.array(hist_steps),
    hist_curves=np.array(hist_curves),
    proj_ctrl=proj_ctrl, ctrl_losses=np.array(
        [lA] + mid_losses + [lB]),
    K=np.array([K]), STEPS=np.array([STEPS]), LR=np.array([LR]),
    meta=np.array(
        f"Garipov piecewise-linear mode connectivity; pair={PAIR}; "
        "A=v5b_s4_seed424243 best (fixed); "
        + ("B=v5b_s3_gtfix_dp best (fixed)" if PAIR == "AB" else
           "B=v5b_s4_seed424245 best (fixed)" if PAIR == "AC" else
           "B=random init (torch.manual_seed(0))") + "; "
        "K=7 ctrl pts (5 mid optimized), Adam lr=1e-3, S=150 steps, "
        "grad clip 5.0; loss=OrderedKeypointLoss total, 50 正样本 "
        "(default_rng(424242), 同 landscape 口径); gp_module=None in "
        "curve_loss (gp 正则在 mid 点无意义, 端点 loss 含 gp≈0)",
        dtype=np.str_))
print("saved mc_data.npz" if not TAG else f"saved mc_data{TAG}.npz")

# ---------- 图 ----------
cmap = LinearSegmentedColormap.from_list(
    "loss_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
cmap.set_bad("#e8e7e2")

# figA：barrier 曲线
fig, ax = plt.subplots(figsize=(9, 5.4))
fig.patch.set_facecolor("white")
ax.plot(ts2, curve_losses, color=C["blue"], lw=2.2,
        label=f"优化后折线路径（max={mc_curve_max:.3f}）")
ax.plot(ts, line_losses, color=C["red"], lw=2.0,
        label=f"直线插值（max={mc_line_max:.3f}）")
ax.axhline(0.1, color=C["mut"], lw=0.8, ls=(0, (4, 3)))
ax.text(0.99, 0.1 * 1.15 + 0.008, "主判据 max<0.1", transform=ax.get_yaxis_transform(),
        ha="right", fontsize=8, color=C["mut"])
ax.set_yscale("log")
ax.set_xlabel(f"路径参数 t（0=A=s424243，1={label_B}）", fontsize=10,
              color=C["sec"])
ax.set_ylabel("loss（log）", fontsize=10, color=C["sec"])
ax.set_title(
    f"图A  Mode connectivity：直线 barrier vs Garipov 折线路径（配对 {PAIR}）\n"
    "（端点 A=真实解固定；50 正样本同 landscape 口径）",
    fontsize=12, color=C["ink"])
ax.grid(True, color=C["grid"], linewidth=0.6, which="both")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(C["axis"])
ax.tick_params(colors=C["mut"], labelsize=9)
ax.legend(fontsize=9, frameon=False, loc="upper center")
fig.tight_layout()
pA = os.path.join(OUT, f"figA_barrier_curve{TAG}.png")
fig.savefig(pA, dpi=110)
plt.close(fig)
print(f"figA -> {pA}")

# figB：α 平面投影
a1s, a2s = z["alphas1"], z["alphas2"]
loss_grid = z["loss_grid"]
A1, A2 = np.meshgrid(a1s, a2s, indexing="ij")
fig, ax = plt.subplots(figsize=(8.6, 7))
fig.patch.set_facecolor("white")
ax.pcolormesh(A1, A2, loss_grid, cmap=cmap,
              vmin=np.nanmin(loss_grid), vmax=np.nanmax(loss_grid),
              shading="nearest")
ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color=C["axis"], lw=0.8,
        ls=(0, (4, 3)), zorder=2)
ax.plot(line_proj[:, 0], line_proj[:, 1], color=C["red"], lw=1.6,
        ls=(0, (4, 3)), zorder=4, label="直线（投影）")
ax.plot(proj_ctrl[:, 0], proj_ctrl[:, 1], color=C["blue"], lw=2.2,
        marker="o", ms=5, zorder=5, label="优化后折线（控制点）")
for i, (px, py) in enumerate(proj_ctrl):
    ax.scatter(px, py, s=40, c=C["blue"], zorder=6,
               edgecolors="white", linewidths=1.0)
ax.scatter(proj_ctrl[0, 0], proj_ctrl[0, 1], s=110, c=C["orange"],
           marker="^", zorder=6, edgecolors="white", linewidths=1.2)
ax.scatter(proj_ctrl[-1, 0], proj_ctrl[-1, 1], s=110, c=C["aqua"],
           marker="s", zorder=6, edgecolors="white", linewidths=1.2)
ax.annotate("A: s424243", proj_ctrl[0], xytext=(-6, -22),
            textcoords="offset points", fontsize=8.5, color=C["ink"],
            ha="right")
ax.annotate(label_B, proj_ctrl[-1], xytext=(-6, 10),
            textcoords="offset points", fontsize=8.5, color=C["ink"],
            ha="right")
ax.set_xlabel("α1（→ seed424243 方向）", fontsize=10, color=C["sec"])
ax.set_ylabel("α2（→ seed424245 方向）", fontsize=10, color=C["sec"])
ax.set_title(
    f"图B  优化后路径投影到 landscape 平面（配对 {PAIR}）：如何绕开高 loss 区\n"
    "（背景=工作项 2 的 21×21 扫描；折线出平面后 z 方向自由度不可见）",
    fontsize=11.5, color=C["ink"])
ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(C["axis"])
ax.tick_params(colors=C["mut"], labelsize=9)
fig.tight_layout()
pB = os.path.join(OUT, f"figB_curve_projection{TAG}.png")
fig.savefig(pB, dpi=110)
plt.close(fig)
print(f"figB -> {pB}")

# figC：优化演化
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")
ax.plot(hist_steps, hist_curves, color=C["aqua"], lw=2, marker="o", ms=3.5)
ax.set_xlabel("优化 step", fontsize=10, color=C["sec"])
ax.set_ylabel("中间点平均 loss", fontsize=10, color=C["sec"])
ax.set_title(
    "图C  路径优化过程：从撞 barrier 到压平\n"
    f"（Adam lr={LR}，{STEPS} 步，每 {hist_every} 步一记）",
    fontsize=12, color=C["ink"])
ax.grid(True, color=C["grid"], linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(C["axis"])
ax.tick_params(colors=C["mut"], labelsize=9)
fig.tight_layout()
pC = os.path.join(OUT, f"figC_loss_evolution{TAG}.png")
fig.savefig(pC, dpi=110)
plt.close(fig)
print(f"figC -> {pC}")

verdict = "PASS" if mc_curve_max < 0.1 else (
    "WEAK" if mc_curve_max < 0.5 else "FAIL")
print(f"主判据：曲线 max={mc_curve_max:.4f} -> {verdict}")
print("DONE round12 工作项4 mode connectivity")
