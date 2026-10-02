# -*- coding: utf-8 -*-
"""Round12 补强 P3：ARI ε 扫描——RI 端点松绑的临界距离。

原 ARI 设计把 RI 端点固定 = 同义反复（端点在 barrier 外，末段必爬台）。
本实验把 RI 端点设为可优化，但约束其权重距原始初始化不超过 ε
（相对权重范数 ||Δθ||/||θ_RI|| ≤ ε），扫描 ε ∈ {0.5,1,2,5,10,20}%：
- 曲线 max 突降的临界 ε = "到达低 loss 通道需要走多远"
- ε=20% 仍高 → 训练必要性论证成立（强）
- ε=1% 就够 → 训练不必要（弱化候选 c）

实现：RI 端点 theta_RI 也进优化器；每步 opt 后做范数投影
（Δ 超限则缩回球面）。中间点不设限。
落盘 mc_ext_eps_scan.npz（每 ε 一个 part）。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "lib"))

import functools
import time
print = functools.partial(print, flush=True)

import numpy as np
import torch
from torch.func import functional_call

from models import create_v5_model
from training.ordered_kp_loss import OrderedKeypointLoss

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_mode_connectivity")
os.makedirs(OUT, exist_ok=True)
VAL_CACHE = os.path.join(PROJECT_ROOT, "logs", "sim_cache",
                         "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
STEPS = 150
LR = 1e-3
N_SAMPLES = 50
K = 7
EPS_LIST = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
print(f"device = {DEV}  eps scan {EPS_LIST}")


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
def eval_sd(params):
    model.load_state_dict(
        {k: (v.detach() if v.requires_grad else v) for k, v in params.items()},
        strict=True)
    model.eval()
    out = model(x1, x2)
    return loss_fn(
        pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
        true_bbox=tb, true_crack_pixels=pk,
        gp_module=None, pixel_mask=pm, true_kpts=tk, kpt_mask=km,
        pred_validity=out.get("validity"),
        true_position_6d=None,
        pred_position_6d=out.get("position_6d"))["total"].item()


results = []
for ei, eps in enumerate(EPS_LIST):
    torch.manual_seed(4242)
    np.random.seed(4242)
    sd_A = load_sd("v5b_s4_seed424243")
    sd_RI = random_init_sd(0)
    theta_A = make_leaf(sd_A)
    theta_RI = make_leaf(sd_RI)
    ri0 = flat_sd_t(sd_RI)                      # RI 原始扁平向量
    ri0_norm = ri0.norm().item()

    mids = [make_leaf(sd_A) for _ in range(1, K - 1)]
    with torch.no_grad():
        for i, p_mid in enumerate(mids, start=1):
            t = i / (K - 1)
            for k in FLOAT_KEYS:
                p_mid[k].copy_(theta_A[k].detach() * (1 - t)
                               + theta_RI[k].detach() * t)

    pts_opt = mids + [theta_RI]                 # RI 端点可优化

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
    diverged = False
    t0 = time.time()
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = curve_loss()
        if not torch.isfinite(loss):
            print(f"[eps={eps:.1%}] step {step}: 发散，停")
            diverged = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
        opt.step()
        # RI 端点范数约束投影：Δθ 超出 ε 球面则缩回
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

    # 优化后路径采样（控制点+段中点）
    @torch.no_grad()
    def sample_path():
        path_pts = [theta_A] + mids + [theta_RI]
        n_ctrl = 2 * (K - 1)
        ts = np.linspace(0.0, 1.0, n_ctrl)
        losses = np.full(n_ctrl, np.nan)
        for j, t in enumerate(ts):
            x = t * (K - 1)
            seg = int(np.clip(np.floor(x), 0, K - 2))
            u = x - seg
            sd_mix = {}
            for k in FLOAT_KEYS:
                sd_mix[k] = (path_pts[seg][k].detach() * (1 - u)
                             + path_pts[seg + 1][k].detach() * u)
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
        return ts, losses

    ts2, curve_losses = sample_path()
    curve_max = float(np.nanmax(curve_losses))
    # RI 端点实际走了多远（范数比）
    with torch.no_grad():
        ri_final = torch.cat(
            [theta_RI[k].detach().reshape(-1) for k in FLOAT_KEYS])
        ri_travel = ((ri_final - ri0).norm() / ri0_norm).item()
    print(f"[eps={eps:.1%}] 曲线 max={curve_max:.4f} "
          f"RI 实际位移={ri_travel*100:.2f}%（上限 {eps*100:.1f}%）")
    results.append(dict(eps=eps, curve_max=curve_max,
                        ri_travel=ri_travel, diverged=diverged,
                        ts_curve=ts2, curve_losses=curve_losses))
    np.savez_compressed(
        os.path.join(OUT, f"mc_ext_epsscan_part{ei}.npz"),
        meta=np.array(
            f"P3 eps scan PARTIAL {ei+1}/{len(EPS_LIST)} "
            f"({time.strftime('%H:%M:%S')}); RI endpoint optimizable "
            f"with ||dtheta||/||ri||<={eps}", dtype=np.str_),
        **{f"r{i}_{k}": v for i, rr in enumerate(results)
           for k, v in rr.items()})

maxes = np.array([r["curve_max"] for r in results])
epss = np.array([r["eps"] for r in results])
crit = np.where(maxes < 0.1)[0]
crit_eps = epss[crit[0]] if len(crit) else None
print(f"P3 VERDICT: eps={np.round(epss,3).tolist()} "
      f"max={np.round(maxes,4).tolist()} "
      f"临界 ε={'%.1f%%' % (crit_eps*100) if crit_eps else '未出现(<0.1)'}")
np.savez_compressed(
    os.path.join(OUT, "mc_ext_epsscan.npz"),
    meta=np.array(
        f"P3 eps scan FINAL; RI endpoint optimizable, "
        f"norm constraint; eps={EPS_LIST}; curve_max={np.round(maxes,4)}"
        f"; crit_eps={crit_eps}", dtype=np.str_),
    **{f"r{i}_{k}": v for i, rr in enumerate(results)
       for k, v in rr.items()},
    eps_list=epss, curve_maxes=maxes)
print("saved mc_ext_epsscan.npz")
print("DONE P3 eps scan")
