# -*- coding: utf-8 -*-
"""MC 诊断：K=7 vs K=15 单步耗时（10 步 forward+backward wall-clock）。

在 K=15 后台任务运行中测量——两个测量同样受当前 GPU 状态扰动，
比值仍可比。落盘 mc_ext_steptime.npz。
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "lib"))

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch
from torch.func import functional_call

from models import create_v5_model
from training.ordered_kp_loss import OrderedKeypointLoss

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_mode_connectivity")


def build_model():
    return create_v5_model("resnet18", image_channels=3, image_size=128,
                           pretrained_2d=False, use_gp=True, min_kpts=8,
                           max_kpts=16, spatial_head=False).to(DEV)


model = build_model()
PARAM_KEYS = set(dict(model.named_parameters()).keys())
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
FLOAT_KEYS = [k for k, v in base_state.items() if v.dtype.is_floating_point]

d = np.load(os.path.join(PROJECT_ROOT, "logs", "sim_cache",
            "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz"))
idx = np.load(os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
              "round12_loss_landscape", "landscape_data.npz"))["sample_idx"]
x1 = torch.from_numpy(d["x_1d"][idx]).float().to(DEV)
x2 = torch.from_numpy(d["x_2d"][idx]).float().to(DEV)
pk = torch.from_numpy(d["true_crack_pixels"][idx]).float().to(DEV)
pm = torch.from_numpy(d["pixel_mask"][idx]).to(DEV)
tk = torch.from_numpy(d["true_keypoints"][idx]).float().to(DEV)
km = torch.from_numpy(d["keypoint_mask"][idx]).to(DEV)
tb = torch.from_numpy(d["true_bbox"][idx]).float().to(DEV)
loss_fn = OrderedKeypointLoss(
    lambda_coverage=1.0, lambda_bbox=1.0, lambda_gp=0.1, lambda_poisson=0.05,
    lambda_kpt_direct=0.5, lambda_ordered=0.0, lambda_validity=0.0,
    lambda_position_6d=0.0, lambda_uv_projection=0.0)


def measure(K, n_steps=10):
    torch.manual_seed(0)
    mids = [{k: (v.detach().clone().to(DEV).requires_grad_(True)
                 if k in PARAM_KEYS else v.detach().clone().to(DEV))
             for k, v in base_state.items()} for _ in range(K - 2)]
    # 初始化到随机权重附近（耗时测量与初值无关，省去端点加载）
    mid_params = [th[k] for th in mids for k in PARAM_KEYS]
    opt = torch.optim.Adam(mid_params, lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        model.eval()
        total = 0.0
        for th in mids:
            out = functional_call(model, th, (x1, x2))
            lo = loss_fn(pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
                         true_bbox=tb, true_crack_pixels=pk, gp_module=None,
                         pixel_mask=pm, true_kpts=tk, kpt_mask=km,
                         pred_validity=out.get("validity"),
                         true_position_6d=None,
                         pred_position_6d=out.get("position_6d"))
            total = total + lo["total"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
        opt.step()

    for _ in range(3):
        step()  # 预热（cudnn benchmark / 分配器）
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_steps
    per_fwd = dt / (K - 2)
    print(f"K={K:2d}: 单步(全 {K-2} mid)={dt*1000:.0f} ms  "
          f"折合每次 mid 前向反向={per_fwd*1000:.0f} ms")
    return dt


dt7 = measure(7)
dt15 = measure(15)
ratio = dt15 / dt7
print(f"K=15/K=7 单步比值 = {ratio:.2f}（理论计算量比 2.6）")
np.savez_compressed(
    os.path.join(OUT, "mc_ext_steptime.npz"),
    dt7=np.array([dt7]), dt15=np.array([dt15]), ratio=np.array([ratio]),
    meta=np.array("10-step forward+backward wall-clock, measured while "
                  "K=15 job running (both measurements equally perturbed)",
                  dtype=np.str_))
print("saved mc_ext_steptime.npz")
print("DONE 单步耗时诊断")
