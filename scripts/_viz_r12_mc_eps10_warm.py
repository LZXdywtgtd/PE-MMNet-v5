# -*- coding: utf-8 -*-
"""Round12 补强 P3b：ε=10% 预热启动控制实验——排除反弹的优化伪影解释。

P3 ε 扫描结果非单调（5%→0.214，10%→0.829，20%→0.742）。可行域
变大最优 loss 应单调不增，违反单调性有两种解释：
1. 几何必然：20% 邻域内确实无低 loss 通道（当前采纳）
2. 优化困难：邻域内有通道但 Adam lr=1e-3 从直线初始化收敛不到

本实验排除解释 2：从 ε=5% 最优解（可行域内已知最好的解）预热
启动 ε=10% 优化。若仍 ≈0.83 → 几何解释成立（强宣称成立）；
若 ≤0.214 → 优化解释成立（P3 弱化为"当前优化方案下 5% 最好"）。

实现：epsscan 未落盘控制点权重，phase1 按 epsscan 完全相同口径
（seed 4242、150 步、lr 1e-3、范数投影）复跑 ε=5% 得到其解；
phase2 用该解初始化 ε=10% 的 mids+RI 端点再跑 150 步。
额外诊断：分别记录控制点 loss 与段中点插值 loss——区分反弹来自
控制点本身还是线性插值段切过 barrier。

落盘 mc_ext_eps10_warm.npz。判据线：warm max ≤0.214 → 优化解释；
≈0.83 → 几何解释。
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
print(f"device = {DEV}")


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


def run_eps_opt(eps, theta_A, theta_RI, mids, ri0_anchor):
    """epsscan 同口径：mids+RI 可优化，RI 范数约束 ≤eps（锚定 ri0_anchor）。"""
    pts_opt = mids + [theta_RI]
    ri0 = ri0_anchor
    ri0_norm = ri0.norm().item()

    def curve_loss():
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
        return total

    mid_params = [th[k] for th in pts_opt for k in PARAM_KEYS]
    opt = torch.optim.Adam(mid_params, lr=LR)
    hist = []
    t0 = time.time()
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = curve_loss()
        if not torch.isfinite(loss):
            print(f"[eps={eps:.1%}] step {step}: 发散，停")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
        opt.step()
        with torch.no_grad():
            ri_now = torch.cat(
                [theta_RI[k].detach().reshape(-1) for k in FLOAT_KEYS])
            delta = ri_now - ri0
            dn = delta.norm().item()
            if dn > eps * ri0_norm:
                scale = eps * ri0_norm / dn
                ptr = 0
                for k in FLOAT_KEYS:
                    n = theta_RI[k].numel()
                    theta_RI[k].sub_(
                        (delta[ptr:ptr + n].reshape(theta_RI[k].shape)
                         * (1 - scale)))
                    ptr += n
        if step % 25 == 0:
            hist.append(loss.item() / (K - 1))
            print(f"[eps={eps:.1%}] step {step:3d} "
                  f"mean={loss.item()/(K-1):.4f} "
                  f"({(time.time()-t0)/60:.1f}min)")
    return hist


@torch.no_grad()
def sample_path(theta_A, theta_RI, mids):
    """控制点与段中点分开记录：ctrl_losses(6) / segmid_losses(6)。"""
    path_pts = [theta_A] + mids + [theta_RI]
    ctrl_losses = np.array([single_loss(th) for th in path_pts])
    segmid_losses = np.array([
        single_loss({k: (path_pts[i][k].detach() * 0.5
                         + path_pts[i + 1][k].detach() * 0.5)
                     for k in FLOAT_KEYS})
        for i in range(len(path_pts) - 1)])
    return ctrl_losses, segmid_losses


# ---------- 公共端点 ----------
torch.manual_seed(4242)
np.random.seed(4242)
sd_A = load_sd("v5b_s4_seed424243")
sd_RI = random_init_sd(0)

# ---------- phase1：复跑 ε=5% 得到其解 ----------
theta_A = make_leaf(sd_A)
theta_RI5 = make_leaf(sd_RI)
mids5 = [make_leaf(sd_A) for _ in range(1, K - 1)]
with torch.no_grad():
    for i, p in enumerate(mids5, start=1):
        t = i / (K - 1)
        for k in FLOAT_KEYS:
            p[k].copy_(theta_A[k].detach() * (1 - t)
                       + theta_RI5[k].detach() * t)
ri0_flat = flat_sd_t(sd_RI)
hist5 = run_eps_opt(0.05, theta_A, theta_RI5, mids5, ri0_flat)
ctrl5, seg5 = sample_path(theta_A, theta_RI5, mids5)
max5 = float(max(ctrl5.max(), seg5.max()))
print(f"[eps=5%] 复跑曲线 max={max5:.4f}（原扫描 0.2143，"
      f"复现差 {abs(max5-0.2143):.4f}）")

# ---------- phase2：ε=10% 预热启动 ----------
theta_RI10 = {k: v.detach().clone() for k, v in theta_RI5.items()}
for k in PARAM_KEYS:
    theta_RI10[k].requires_grad_(True)
mids10 = [{k: v.detach().clone() for k, v in th.items()} for th in mids5]
for th in mids10:
    for k in PARAM_KEYS:
        th[k].requires_grad_(True)
hist10 = run_eps_opt(0.10, theta_A, theta_RI10, mids10, ri0_flat)
ctrl10, seg10 = sample_path(theta_A, theta_RI10, mids10)
max10_warm = float(max(ctrl10.max(), seg10.max()))
ri0n = ri0_flat.norm().item()
with torch.no_grad():
    ri_travel = float((flat_sd_t({k: theta_RI10[k].detach()
                                  for k in FLOAT_KEYS})
                       - flat_sd_t(sd_RI)).norm().item() / ri0n)
print(f"[eps=10% warm] 曲线 max={max10_warm:.4f}"
      f"（冷启动对照 0.8287）  RI 实际位移={ri_travel*100:.2f}%")
print(f"  控制点 loss: {np.round(ctrl10, 4).tolist()}")
print(f"  段中点 loss: {np.round(seg10, 4).tolist()}")

# ---------- 判据 ----------
if max10_warm <= 0.214:
    verdict = ("优化解释成立：预热把 10% 压到 5% 水平 -> P3 结论弱化为"
               "\"当前优化方案下 5% 最好\"")
elif max10_warm >= 0.5:
    verdict = ("几何解释成立：从 5% 最优解预热仍反弹 -> 10-20% 邻域内"
               "无低 loss 通道（或通道不在 Adam 可达方向），强宣称保持")
else:
    verdict = "中间结果：预热有改善但未达 5% 水平——两种解释各占一部分"
print(f"P3b VERDICT: warm max={max10_warm:.4f} vs cold 0.8287 / "
      f"eps5 {max5:.4f}\n  -> {verdict}")

np.savez_compressed(
    os.path.join(OUT, "mc_ext_eps10_warm.npz"),
    eps5_curve_max=max5, eps5_ctrl_losses=ctrl5, eps5_segmid_losses=seg5,
    warm_curve_max=max10_warm, warm_ctrl_losses=ctrl10,
    warm_segmid_losses=seg10, ri_travel=ri_travel,
    hist_eps5=np.array(hist5), hist_warm=np.array(hist10),
    meta=np.array(
        "P3b control: eps=10% warm-started from eps=5% solution "
        "(re-run, seed 4242, same protocol as epsscan). If warm max "
        "still ~0.83 -> bounce is geometric, not optimization "
        "artifact. cold-start eps10 reference = 0.8287", dtype=np.str_))
print("saved mc_ext_eps10_warm.npz")
print("DONE P3b eps10 warm")
