# -*- coding: utf-8 -*-
"""Round12 补强 P2：稳定性强化——2a 十 seed AB + 2b 五个 RI 重复。

2a：AB K=7，opt_seed 42/137/2024（已有）+ 7/99/555/1000/3141/9999/2718，
    报 mean/std/CV/min/max。判据 CV<10%。
2b：ARI K=7，RI 端点 seed 0/1/2/3/4 各跑一次。判据全部 FAIL（max>0.5），
    支持"ARI barrier 不是单点偶然"。

口径与主实验一致：50 正样本、端点固定、Adam lr=1e-3、150 步、clip 5.0。
分阶段落盘 mc_ext_10seed_part{j}.npz / mc_ext_ari5ri_part{j}.npz。
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
print(f"device = {DEV}  P2 stability (batch = {N_SAMPLES})")

SEEDS_10 = [42, 137, 2024, 7, 99, 555, 1000, 3141, 9999, 2718]
RI_SEEDS = [0, 1, 2, 3, 4]


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


def run_mc(pair, K, opt_seed, tag, ri_seed=0):
    torch.manual_seed(opt_seed)
    np.random.seed(opt_seed % (2**31))
    sd_A = load_sd("v5b_s4_seed424243")
    if pair == "AB":
        sd_B, label_B = load_sd("v5b_s3_gtfix_dp"), "dp"
    else:
        sd_B, label_B = random_init_sd(ri_seed), f"RI(seed{ri_seed})"
    theta_A = make_leaf(sd_A)
    theta_B = make_leaf(sd_B)
    with torch.no_grad():
        lA = eval_sd(theta_A)
        lB = eval_sd(theta_B)
    mids = [make_leaf(sd_A) for _ in range(1, K - 1)]
    with torch.no_grad():
        for i, p_mid in enumerate(mids, start=1):
            t = i / (K - 1)
            for k in FLOAT_KEYS:
                p_mid[k].copy_(theta_A[k].detach() * (1 - t)
                               + theta_B[k].detach() * t)

    def path_points():
        return [theta_A] + mids + [theta_B]

    def curve_loss():
        model.eval()
        total = 0.0
        for th in path_points()[1:-1]:
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
    def sample_path_losses(n_pts=100):
        ts = np.linspace(0.0, 1.0, n_pts)
        losses = np.full(n_pts, np.nan)
        pts = path_points()
        for j, t in enumerate(ts):
            x = t * (K - 1)
            seg = int(np.clip(np.floor(x), 0, K - 2))
            u = x - seg
            sd_mix = {}
            for k in FLOAT_KEYS:
                sd_mix[k] = (pts[seg][k].detach() * (1 - u)
                             + pts[seg + 1][k].detach() * u)
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

    ts_line, line_losses = sample_path_losses()

    mid_params = [th[k] for th in mids for k in PARAM_KEYS]
    opt = torch.optim.Adam(mid_params, lr=LR)
    hist_steps, hist_curves = [], []
    t_start = time.time()
    diverged = False
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = curve_loss()
        if not torch.isfinite(loss):
            print(f"[{tag}] step {step}: NaN/Inf 发散，停")
            diverged = True
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
        opt.step()
        if step % 25 == 0:
            hist_steps.append(step)
            hist_curves.append(loss.item() / (K - 2))
            el = time.time() - t_start
            eta = el / (step + 1) * (STEPS - step - 1)
            print(f"[{tag}] step {step:3d} mean_mid={loss.item()/(K-2):.4f} "
                  f"({el/60:.1f}min, ETA {eta/60:.1f}min)")

    n_ctrl = 2 * (K - 1)
    ts2, curve_losses = sample_path_losses(n_pts=n_ctrl)
    curve_max = float(np.nanmax(curve_losses))
    print(f"[{tag}] 端点 A={lA:.4f} B({label_B})={lB:.4f} | "
          f"曲线 max={curve_max:.4f}")
    return dict(pair=pair, K=K, opt_seed=opt_seed, ri_seed=ri_seed,
                lA=lA, lB=lB, diverged=diverged,
                ts_curve=ts2, curve_losses=curve_losses,
                hist_steps=np.array(hist_steps),
                hist_curves=np.array(hist_curves),
                curve_max=curve_max)


# ---------- 2a：10 seed AB ----------
t0 = time.time()
r10 = []
for ji, s in enumerate(SEEDS_10):
    r = run_mc("AB", 7, s, f"AB-s{s}")
    r10.append(r)
    np.savez_compressed(
        os.path.join(OUT, f"mc_ext_10seed_part{ji}.npz"),
        meta=np.array(
            f"P2a 10seed AB K=7 PARTIAL {ji+1}/{len(SEEDS_10)} "
            f"({time.strftime('%H:%M:%S')})", dtype=np.str_),
        **{f"r{i}_{k}": v for i, rr in enumerate(r10)
           for k, v in rr.items()})
    print(f">>> 2a 阶段落盘 {ji+1}/{len(SEEDS_10)}")

maxes = np.array([r["curve_max"] for r in r10])
mean, std = maxes.mean(), maxes.std(ddof=1)
cv = std / mean * 100
print(f"2a VERDICT 10seed: mean={mean:.4f} std={std:.4f} CV={cv:.1f}% "
      f"min={maxes.min():.4f} max={maxes.max():.4f}")
np.savez_compressed(
    os.path.join(OUT, "mc_ext_10seed.npz"),
    meta=np.array(
        f"P2a AB K=7 10 opt_seeds {SEEDS_10}; mean={mean:.4f} "
        f"std={std:.4f} CV={cv:.1f}%; 50 pos; Adam 1e-3 S=150",
        dtype=np.str_),
    **{f"r{i}_{k}": v for i, rr in enumerate(r10)
       for k, v in rr.items()},
    mean=np.array([mean]), std=np.array([std]), cv=np.array([cv]))
print(f"saved mc_ext_10seed.npz ({(time.time()-t0)/60:.1f}min)")
print("DONE P2a 10seed")

# ---------- 2b：5 个 RI 重复 ----------
t0 = time.time()
r5 = []
for ji, s in enumerate(RI_SEEDS):
    r = run_mc("ARI", 7, 4242, f"ARI-ri{s}", ri_seed=s)
    r5.append(r)
    np.savez_compressed(
        os.path.join(OUT, f"mc_ext_ari5ri_part{ji}.npz"),
        meta=np.array(
            f"P2b 5 RI seeds ARI K=7 PARTIAL {ji+1}/{len(RI_SEEDS)} "
            f"({time.strftime('%H:%M:%S')})", dtype=np.str_),
        **{f"r{i}_{k}": v for i, rr in enumerate(r5)
           for k, v in rr.items()})
    print(f">>> 2b 阶段落盘 {ji+1}/{len(RI_SEEDS)}")

maxes5 = np.array([r["curve_max"] for r in r5])
n_fail = int((maxes5 > 0.5).sum())
print(f"2b VERDICT 5RI: maxes={np.round(maxes5,4).tolist()} "
      f"FAIL 数={n_fail}/5（判据全部>0.5）")
np.savez_compressed(
    os.path.join(OUT, "mc_ext_ari5ri.npz"),
    meta=np.array(
        f"P2b ARI K=7 with RI endpoint seeds {RI_SEEDS}; "
        f"curve_max per RI={np.round(maxes5,4).tolist()}; "
        f"{n_fail}/5 FAIL(>0.5); 50 pos; Adam 1e-3 S=150",
        dtype=np.str_),
    **{f"r{i}_{k}": v for i, rr in enumerate(r5)
       for k, v in rr.items()},
    ri_curve_max=maxes5)
print(f"saved mc_ext_ari5ri.npz ({(time.time()-t0)/60:.1f}min)")
print("DONE P2b 5RI")
