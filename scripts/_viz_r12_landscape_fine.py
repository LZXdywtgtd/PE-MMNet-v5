# -*- coding: utf-8 -*-
"""Round12 工作项 2f：landscape 加密扫描（有景观子域）。

v4 全域 21×21（[-0.1,1.1]²）确认有景观支撑集中在 [-0.1,0.5]² 子域
（153/441 finite 且低 loss 三角形全在此内）。本脚本同口径加密：
51×51 覆盖 [-0.1,0.5]²（2601 格 × 50 样本单 batch）。

平面定义与 _viz_r12_landscape.py 完全一致：
center=v5b_s3_gtfix_dp best 全 state；d1→s424243、d2→s424245 全 state
差向量；BN buffer 一并插值；num_batches_tracked 固定。
λ = run_train_v5 默认（cov 1.0/bbox 1.0/gp 0.1/poisson 0.05/kpt_direct 0.5）
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

LAND_OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                        "round12_loss_landscape")
VAL_CACHE = os.path.join(PROJECT_ROOT, "logs", "sim_cache",
                         "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
CHAMP = "v5b_s3_gtfix_dp"
AXIS_RUNS = ["v5b_s4_seed424243", "v5b_s4_seed424245"]
N_GRID = 51
A_LO, A_HI = -0.1, 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {DEV}")


def build_model():
    return create_v5_model(
        "resnet18", image_channels=3, image_size=128, pretrained_2d=False,
        use_gp=True, min_kpts=8, max_kpts=16, spatial_head=False).to(DEV)


def flat_state(model):
    return np.concatenate(
        [v.detach().double().cpu().numpy().ravel()
         for v in model.state_dict().values()
         if v.dtype.is_floating_point])


def state_to_flat_dict(flat, base_state):
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

model = build_model()
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def eval_flat(flat):
    try:
        model.load_state_dict(state_to_flat_dict(flat, base_state))
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


def load_flat(run):
    if run == "__random_init__":
        return flat_state(build_model())
    load_checkpoint(os.path.join(TH, run, "checkpoints", "best.pt"),
                    model, map_location=DEV)
    return flat_state(model)


theta_c = load_flat(CHAMP)
M = np.stack([load_flat(AXIS_RUNS[0]) - theta_c,
              load_flat(AXIS_RUNS[1]) - theta_c])
t0, c0 = eval_flat(theta_c)
print(f"center 本体 loss={t0:.4f}（应≈0.0157）")

a1s = np.linspace(A_LO, A_HI, N_GRID)
a2s = np.linspace(A_LO, A_HI, N_GRID)
loss_grid = np.full((N_GRID, N_GRID), np.nan)
cov_grid = np.full((N_GRID, N_GRID), np.nan)
done = 0
for i, b1 in enumerate(a1s):
    for j, b2 in enumerate(a2s):
        loss_grid[i, j], cov_grid[i, j] = eval_flat(
            theta_c + b1 * M[0] + b2 * M[1])
    done += N_GRID
    print(f"[{done}/{N_GRID*N_GRID}] row {i+1}: "
          f"finite {np.isfinite(loss_grid[i]).sum()}/{N_GRID}")

np.savez_compressed(
    os.path.join(LAND_OUT, "landscape_data_fine.npz"),
    alphas1=a1s, alphas2=a2s,
    loss_grid=loss_grid, cov_grid=cov_grid,
    center_loss=np.array([t0]),
    sample_idx=sample_idx,
    meta=np.array(
        "fine grid 51x51 [-0.1,0.5]^2 (有景观子域加密, 同 v4 平面口径); "
        "plane: center=v5b_s3_gtfix_dp best full-state; d1=→s424243, "
        "d2=→s424245; lambda: cov1.0 bbox1.0 gp0.1 poisson0.05 "
        "kpt_direct0.5; loss=OrderedKeypointLoss total, 50 正样本单 batch "
        "(default_rng(424242))", dtype=np.str_))
sz = os.path.getsize(os.path.join(LAND_OUT, "landscape_data_fine.npz")) / 1e6
print(f"saved landscape_data_fine.npz ({sz:.3f} MB)")
print(f"grid loss: finite {np.isfinite(loss_grid).sum()}/{loss_grid.size}, "
      f"range [{np.nanmin(loss_grid):.4f}, {np.nanmax(loss_grid):.4f}]")
print("DONE round12 工作项2f 加密扫描")
