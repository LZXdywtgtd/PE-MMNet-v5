# -*- coding: utf-8 -*-
"""Round12 补强 P3c：ε=10% 迭代预热——插值段残余 barrier 是否可被
持续优化压掉。

P3b 单次预热：冷启动 0.8287 -> warm 0.2450（-70%），残余 max 全在
末段插值中点。遗留问题（审稿人必问）："预热一次不够，多预热几次
是否就能压到盆地（0.1）？"

本实验：ε=10% 约束下连续迭代 5 轮，每轮 150 步、Adam 重新初始化
（动量清零，权重延续上一轮终点），范数约束始终锚定原始 RI。
判据：
- max 持续下降至 <0.15 -> 优化困难主导，"几何必然"站不住
- max 卡在 ~0.2（±0.05）-> 插值段不可压缩是真几何结论，
  候选 c 的两层承载表述成立

落盘 mc_ext_eps10_iter.npz（每轮 part 断点续跑 + 最终合并）。
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
VAL_CACHE = os.path.join(PROJECT_ROOT, "logs", "sim_cache",
                         "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
STEPS = 150
LR = 1e-3
N_SAMPLES = 50
K = 7
EPS = 0.10
N_ROUNDS = 5
print(f"device = {DEV}  iter warmup eps={EPS:.0%} x {N_ROUNDS} rounds")


def build_model():
    return create_v5_model(
        "resnet18", image_channels=3, image_size=128, pretrained_2d=False,
        use_gp=True, min_kpts=8, max_kpts=16, spatial_head=False).to(DEV)


def load_sd(run):
    ck = torch.load(os.path.join(TH, run, "checkpoints", "best.pt"),
                    map_location="cpu", weights_only=False)
    return ck.get("model_state_dict", ck)


model = build_model()
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
FLOAT_KEYS = [k for k, v in base_state.items() if v.dtype.is_floating_point]
PARAM_KEYS = set(dict(model.named_parameters()).keys())

z = np.load(os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                         "round12_loss_landscape", "landscape_data.npz"),
            allow_pickle=False)
sample_idx = z["sample_idx"][:N_SAMPLES]
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


def random_init_sd(seed=0):
    torch.manual_seed(seed)
    m = build_model()
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def flat_sd_t(sd):
    return torch.cat([sd[k].detach().reshape(-1) for k in FLOAT_KEYS])


@torch.no_grad()
def single_loss(th):
    out = functional_call(model, th, (x1, x2))
    return loss_fn(
        pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
        true_bbox=tb, true_crack_pixels=pk,
        gp_module=None, pixel_mask=pm, true_kpts=tk, kpt_mask=km,
        pred_validity=out.get("validity"),
        true_position_6d=None,
        pred_position_6d=out.get("position_6d"))["total"].item()


@torch.no_grad()
def project_ri(theta_RI, ri0, eps):
    ri_now = torch.cat(
        [theta_RI[k].detach().reshape(-1) for k in FLOAT_KEYS])
    delta = ri_now - ri0
    dn = delta.norm().item()
    if dn > eps * ri0.norm().item():
        scale = eps * ri0.norm().item() / dn
        ptr = 0
        for k in FLOAT_KEYS:
            n = theta_RI[k].numel()
            theta_RI[k].sub_(
                (delta[ptr:ptr + n].reshape(theta_RI[k].shape)
                 * (1 - scale)))
            ptr += n


@torch.no_grad()
def sample_path(theta_A, theta_RI, mids):
    path_pts = [theta_A] + mids + [theta_RI]
    ctrl_losses = np.array([single_loss(th) for th in path_pts])
    segmid_losses = np.array([
        single_loss({k: (path_pts[i][k].detach() * 0.5
                         + path_pts[i + 1][k].detach() * 0.5)
                     for k in FLOAT_KEYS})
        for i in range(len(path_pts) - 1)])
    return ctrl_losses, segmid_losses


# ---------- 初始化（同 P3b 修正版口径）----------
torch.manual_seed(4242)
np.random.seed(4242)
sd_A = load_sd("v5b_s4_seed424243")
sd_RI = random_init_sd(0)
theta_A = make_leaf(sd_A)
theta_RI = make_leaf(sd_RI)
mids = [make_leaf(sd_A) for _ in range(1, K - 1)]
with torch.no_grad():
    for i, p in enumerate(mids, start=1):
        t = i / (K - 1)
        for k in FLOAT_KEYS:
            p[k].copy_(theta_A[k].detach() * (1 - t)
                       + theta_RI[k].detach() * t)
ri0 = flat_sd_t(sd_RI)

pts_opt = mids + [theta_RI]
ctrl_h, seg_h, max_h = [], [], []
t0 = time.time()
for rnd in range(1, N_ROUNDS + 1):
    part_path = os.path.join(OUT, f"mc_ext_eps10_iter_part{rnd}.npz")
    if os.path.exists(part_path):
        dd = np.load(part_path)
        ctrl_h = list(dd["ctrl_maxes"]); seg_h = list(dd["seg_maxes"])
        max_h = list(dd["curve_maxes"])
        # 权重状态不可序列化恢复，断点只恢复记录不恢复权重——
        # 若中断发生在轮次中间，后续轮次作废重跑（本实验总时长短）
        print(f"[round {rnd}] part 已存在，跳过（记录级续跑）")
        continue
    # 每轮 Adam 重新初始化（动量清零），权重延续
    mid_params = [th[k] for th in pts_opt for k in PARAM_KEYS]
    opt = torch.optim.Adam(mid_params, lr=LR)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        model.eval()
        total = 0.0
        for th in pts_opt:
            out = functional_call(model, th, (x1, x2))
            lo = loss_fn(
                pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
                true_bbox=tb, true_crack_pixels=pk,
                gp_module=None, pixel_mask=pm, true_kpts=tk, kpt_mask=km,
                pred_validity=out.get("validity"),
                true_position_6d=None,
                pred_position_6d=out.get("position_6d"))
            total = total + lo["total"]
        if not torch.isfinite(total):
            print(f"[round {rnd}] step {step}: 发散，停")
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
        opt.step()
        project_ri(theta_RI, ri0, EPS)
        if step % 50 == 0:
            print(f"[round {rnd}] step {step:3d} "
                  f"mean={total.item()/(K-1):.4f} "
                  f"({(time.time()-t0)/60:.1f}min)")
    ctrl, seg = sample_path(theta_A, theta_RI, mids)
    cmax, smax = float(ctrl.max()), float(seg.max())
    ctrl_h.append(cmax); seg_h.append(smax)
    max_h.append(max(cmax, smax))
    np.savez_compressed(part_path, ctrl_losses=ctrl, segmid_losses=seg,
                        ctrl_maxes=np.array(ctrl_h),
                        seg_maxes=np.array(seg_h),
                        curve_maxes=np.array(max_h))
    print(f"[round {rnd}] ctrl max={cmax:.4f}  seg max={smax:.4f}  "
          f"curve max={max(cmax, smax):.4f}  <<< 轮 {rnd}/{N_ROUNDS} "
          f"({(time.time()-t0)/60:.1f}min)")

print("\n===== 迭代预热轨迹 =====")
for r, m in enumerate(max_h, start=1):
    print(f"round {r}: curve max={m:.4f} (ctrl {ctrl_h[r-1]:.4f} / "
          f"seg {seg_h[r-1]:.4f})")
final = max_h[-1]
if final < 0.15:
    verdict = ("优化困难主导：迭代持续下降至盆地边缘 -> '几何必然'站不住，"
               "P3 反弹主要是冷启动收敛问题")
elif final <= 0.25:
    verdict = ("插值段不可压缩是真几何结论：多轮预热后 max 卡在 ~0.2 -> "
               "候选 c 两层承载表述（起点定位 + 插值几何）成立")
else:
    verdict = "未收敛也未下降：需检查是否陷入别的高 loss 区"
print(f"P3c VERDICT: rounds max={np.round(max_h,4).tolist()} "
      f"final={final:.4f}\n  -> {verdict}")

np.savez_compressed(
    os.path.join(OUT, "mc_ext_eps10_iter.npz"),
    ctrl_maxes=np.array(ctrl_h), seg_maxes=np.array(seg_h),
    curve_maxes=np.array(max_h),
    meta=np.array(
        f"P3c iterated warmup: eps=10% x {N_ROUNDS} rounds, each round "
        "150 steps fresh Adam (momentum reset, weights carried), norm "
        "constraint anchored at original RI. cold ref 0.8287, single "
        "warm ref 0.2450 (P3b). If max plateaus ~0.2 -> segment "
        "incompressibility is real geometry", dtype=np.str_))
print("saved mc_ext_eps10_iter.npz")
print("DONE P3c iter warmup")
