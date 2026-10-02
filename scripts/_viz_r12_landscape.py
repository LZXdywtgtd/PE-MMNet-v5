# -*- coding: utf-8 -*-
"""Round12 工作项 2：loss landscape 扫描（v3 模式连通平面）。

历史教训（写死防回归）：
- v1：扰动坐标含 BN buffer + 扫描范围拉到随机初始化距离（原始坐标
  ~6 万单位）→ buffer/权重失配，441 格 439 NaN。
- v2：filter 归一化方向（Li 2018）破坏锚点重建（归一化改变方向形状，
  2D 重建误差 63-95%），网格里锚点落点 loss 全 1.09=假景观；
  另发现 6 收敛点逐参数均值 loss=1.10（均值不是解 → seed 间存在
  线性插值 barrier，盆地真分离——这本身是结论）。

v3 设计（数学精确）：
- 平面中心 = 收官 run（v5b_s3_gtfix_dp）best.pt 全 state（float 参数
  +BN buffer 一并插值，两端点 state 精确 → 端点 loss=本体 loss）；
  num_batches_tracked（int，不影响 eval 前向）固定
- 两轴 d1/d2 = 到 s424243/s424245 best.pt 的全 state 差向量 →
  (0,0)/(1,0)/(0,1) 三锚点在平面上精确重建，其余锚点最小二乘投影
- 网格 α∈[-1.5,1.5]²，21×21；1D 剖面 dp↔{base,s244,s246,随机初始化}
  α∈[-0.25,1.25]（随机初始化剖面发散段如实 NaN=发散带）
- λ = run_train_v5 默认（cov 1.0/bbox 1.0/gp 0.1/poisson 0.05/
  kpt_direct 0.5）；50 正样本 default_rng(424242) 单 batch 前向
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

OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_loss_landscape")
os.makedirs(OUT, exist_ok=True)

VAL_CACHE = os.path.join(PROJECT_ROOT, "logs", "sim_cache",
                         "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
CHAMP = "v5b_s3_gtfix_dp"          # 平面中心
AXIS_RUNS = ["v5b_s4_seed424243",  # d1
             "v5b_s4_seed424245"]  # d2
CUT_RUNS = ["v5b_s3_gtfix_base", "v5b_s4_seed424244",
            "v5b_s4_seed424246", "__random_init__"]
PROJ_RUNS = ANCHOR_RUNS = ["v5b_s3_gtfix_base", "v5b_s4_seed424243",
                           "v5b_s4_seed424244", "v5b_s4_seed424245",
                           "v5b_s4_seed424246", "__random_init__"]
N_GRID = 21
N_CUT = 23
N_SAMPLES = 50
SEED = 424242
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEV}")


def build_model(variant="resnet18"):
    kw = dict(image_channels=3, image_size=128, pretrained_2d=False,
              use_gp=True, min_kpts=8, max_kpts=16)
    if variant == "resnet18":
        kw["spatial_head"] = False
    return create_v5_model(variant, **kw)


def flat_state(model):
    """全 state 展平（float 项；int 项跳过，eval 前向不受影响）。"""
    return np.concatenate(
        [v.detach().double().cpu().numpy().ravel()
         for v in model.state_dict().values()
         if v.dtype.is_floating_point])


def state_to_flat_dict(model, flat, base_state):
    """flat 向量 → state_dict（float 项按序填回，int 项取 base）。"""
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
    return sd


# ---------- 数据 ----------
d = np.load(VAL_CACHE)
pos_all = np.where(d["pixel_mask"].sum(axis=1) > 0)[0]
rng = np.random.default_rng(SEED)
sample_idx = np.sort(rng.choice(pos_all, size=N_SAMPLES, replace=False))
np.save(os.path.join(OUT, "sample_idx.npy"), sample_idx)
print(f"val 正样本 {len(pos_all)}，抽 {N_SAMPLES}（default_rng({SEED})）")

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

model = build_model().to(DEV)
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def eval_flat(flat):
    try:
        model.load_state_dict(state_to_flat_dict(model, flat, base_state))
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
        t, c = lo["total"].item(), lo["coverage"].item()
        return (t if np.isfinite(t) else np.nan,
                c if np.isfinite(c) else np.nan)
    except Exception:
        return np.nan, np.nan


# ---------- 锚点 state ----------
def load_flat(run):
    if run == "__random_init__":
        m = build_model().to(DEV)
        return flat_state(m)
    load_checkpoint(os.path.join(TH, run, "checkpoints", "best.pt"),
                    model, map_location=DEV)
    return flat_state(model)


theta_c = load_flat(CHAMP)
print(f"center {CHAMP}  |θ|={np.linalg.norm(theta_c):.1f}")
t0, c0 = eval_flat(theta_c)
print(f"center 本体 loss={t0:.4f}（应≈0.0157）")

theta_d1 = load_flat(AXIS_RUNS[0])
theta_d2 = load_flat(AXIS_RUNS[1])
d1 = theta_d1 - theta_c
d2 = theta_d2 - theta_c
t_a1, _ = eval_flat(theta_d1)
t_a2, _ = eval_flat(theta_d2)
print(f"axis1 {AXIS_RUNS[0]} 端点 loss={t_a1:.4f}  "
      f"axis2 {AXIS_RUNS[1]} 端点 loss={t_a2:.4f}")

# ---------- 其余锚点平面坐标（最小二乘投影）----------
M = np.stack([d1, d2])           # (2, D)
G = M @ M.T
Ginv = np.linalg.inv(G)
anchor_coord, anchor_loss = {}, {}
for run in PROJ_RUNS:
    th = load_flat(run)
    a = (th - theta_c) @ M.T @ Ginv
    tl, _ = eval_flat(th)
    anchor_coord[run] = a
    anchor_loss[run] = tl
    print(f"anchor {run:22s} α=({a[0]:+.3f},{a[1]:+.3f})  "
          f"本体 loss={tl:.4f}")

# ---------- 2D 网格 ----------
# 范围探针（2026-10-02）：α1/α2∈[0,1]³ 三角形内有景观（三低角+高原），
# 负方向与 >1.1 外推区大面积发散（NaN=真实景观边界，如实呈现）
A_LO, A_HI = -0.1, 1.1
a1s = np.linspace(A_LO, A_HI, N_GRID)
a2s = np.linspace(A_LO, A_HI, N_GRID)
loss_grid = np.full((N_GRID, N_GRID), np.nan)
cov_grid = np.full((N_GRID, N_GRID), np.nan)
done = 0
for i, b1 in enumerate(a1s):
    for j, b2 in enumerate(a2s):
        loss_grid[i, j], cov_grid[i, j] = eval_flat(
            theta_c + b1 * d1 + b2 * d2)
        done += 1
    print(f"[{done}/{N_GRID*N_GRID}] row {i+1}: "
          f"finite {np.isfinite(loss_grid[i]).sum()}/{N_GRID}")

# ---------- 1D 剖面 ----------
cut_runs, cut_ts, cut_losses, cut_covs = [], [], [], []
for run in AXIS_RUNS + CUT_RUNS:
    th = load_flat(run)
    dd = th - theta_c
    ts = np.linspace(-0.25, 1.25, N_CUT)
    ls = np.full(N_CUT, np.nan)
    cs = np.full(N_CUT, np.nan)
    for i, t in enumerate(ts):
        ls[i], cs[i] = eval_flat(theta_c + t * dd)
    cut_runs.append(run)
    cut_ts.append(ts)
    cut_losses.append(ls)
    cut_covs.append(cs)
    print(f"cut {run:22s} finite {np.isfinite(ls).sum()}/{N_CUT}  "
          f"min={np.nanmin(ls):.4f} max={np.nanmax(ls):.4f}")

# ---------- swin_yolo 量级参照 ----------
swin_loss = np.nan
try:
    swin = build_model("swin_yolo").to(DEV)
    load_checkpoint(os.path.join(TH, "v5b_s5_swin_yolo", "checkpoints",
                                 "best.pt"), swin, map_location=DEV)
    swin.eval()
    with torch.no_grad():
        out = swin(x1, x2)
        lo = loss_fn(
            pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
            true_bbox=tb, true_crack_pixels=pk,
            gp_module=getattr(swin, "gp_module", None),
            pixel_mask=pm, true_kpts=tk, kpt_mask=km,
            pred_validity=out.get("validity"),
            true_position_6d=None,
            pred_position_6d=out.get("position_6d"))
    swin_loss = lo["total"].item()
    print(f"swin_yolo 参照 loss = {swin_loss:.4f}")
except Exception as e:
    print(f"swin_yolo 参照失败（不阻塞）: {e}")

# ---------- 落盘 ----------
labels = [CHAMP] + AXIS_RUNS + CUT_RUNS
np.savez_compressed(
    os.path.join(OUT, "landscape_data.npz"),
    alphas1=a1s, alphas2=a2s,
    loss_grid=loss_grid, cov_grid=cov_grid,
    center_loss=np.array([t0]),
    cut_runs=np.array(cut_runs, dtype=np.str_),
    cut_t=np.stack(cut_ts), cut_loss=np.stack(cut_losses),
    cut_cov=np.stack(cut_covs),
    anchor_runs=np.array(PROJ_RUNS, dtype=np.str_),
    anchor_coord=np.stack([anchor_coord[r] for r in PROJ_RUNS]),
    anchor_loss=np.array([anchor_loss[r] for r in PROJ_RUNS]),
    sample_idx=sample_idx,
    swin_loss=np.array([swin_loss]),
    meta=np.array(
        "plane: center=v5b_s3_gtfix_dp best full-state; "
        "d1=→s424243, d2=→s424245 (全 state 差向量, BN buffer 一并插值); "
        "grid [-0.1,1.1]²（探针确定的有景观支撑区，负方向/外推区发散="
        "真实景观边界，NaN 如实呈现）; cuts α∈[-0.25,1.25]; "
        "对角 buffer 固定对照已验证 barrier 在参数本身非插值伪影; "
        "lambda: cov1.0 bbox1.0 gp0.1 poisson0.05 kpt_direct0.5 "
        "(run_train_v5 默认); loss=OrderedKeypointLoss total, "
        "50 正样本单 batch (default_rng(424242))",
        dtype=np.str_))
sz = os.path.getsize(os.path.join(OUT, "landscape_data.npz")) / 1e6
print(f"saved landscape_data.npz ({sz:.3f} MB)")
print(f"grid loss: finite {np.isfinite(loss_grid).sum()}/{loss_grid.size}, "
      f"range [{np.nanmin(loss_grid):.4f}, {np.nanmax(loss_grid):.4f}]")
print("DONE round12 工作项2 扫描 v3")
