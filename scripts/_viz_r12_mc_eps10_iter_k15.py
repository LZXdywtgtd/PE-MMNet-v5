# -*- coding: utf-8 -*-
"""Round12 补强 P3c-2：K=15 迭代预热——末段擦台是几何固有还是
K=7 折线分辨率伪影，二选一收口。

P3c（K=7）结论：ε=10% 下 5 轮迭代不收敛（0.954/0.407/0.504/
0.726/0.549），控制点全程盆内、反弹全部在段中点、末 2 段擦台
5 轮固定。遗留问题：加密折线（K=15，14 段）后擦台段变短——
若擦台消失/入盆地 -> K=7 擦台是分辨率伪影；若末段仍擦台 ->
几何固有（P3 链条收口）。

协议与 P3c 完全对齐：ε=10% × 5 轮、每轮 150 步、Adam 动量重置、
权重延续、范数锚定原始 RI；唯一差异 K=7->15、batch 50->10
（K=15 batch50 显存不可行的既有结论；batch10 健康门控 3.53 GiB
已验证）。**注意口径**：batch 10 与 P3c batch 50 数字不可直接
比 loss 绝对值，只比"末段 vs 前段"的相对结构与判定翻转与否。

落盘 mc_ext_eps10_iter_k15.npz（每轮 part 断点续跑 + 合并）。
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
N_SAMPLES = 10          # batch 10：K=15 显存约束（同 k15ari 口径）
K = 15
EPS = 0.10
N_ROUNDS = 5
print(f"device = {DEV}  K={K}  batch={N_SAMPLES}  "
      f"iter warmup eps={EPS:.0%} x {N_ROUNDS} rounds")


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
    """控制点 + 段中点（14 控制点 + 14 段中点 = 28 采样点）。"""
    path_pts = [theta_A] + mids + [theta_RI]
    ctrl_losses = np.array([single_loss(th) for th in path_pts])
    segmid_losses = np.array([
        single_loss({k: (path_pts[i][k].detach() * 0.5
                         + path_pts[i + 1][k].detach() * 0.5)
                     for k in FLOAT_KEYS})
        for i in range(len(path_pts) - 1)])
    return ctrl_losses, segmid_losses


# ---------- 初始化（同 P3c 口径：RI seed 0）----------
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
    part_path = os.path.join(OUT, f"mc_ext_eps10_iter_k15_part{rnd}.npz")
    if os.path.exists(part_path):
        dd = np.load(part_path)
        ctrl_h = list(dd["ctrl_maxes"]); seg_h = list(dd["seg_maxes"])
        max_h = list(dd["curve_maxes"])
        print(f"[round {rnd}] part 已存在，跳过（记录级续跑）")
        continue
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
        if rnd == 1 and step == 10:
            mem = torch.cuda.max_memory_allocated() / 2**30
            el = (time.time() - t0)
            print(f"[健康检查] step10: 峰值显存 {mem:.2f} GiB, "
                  f"前10步均耗时 {el/10:.2f}s")
            if mem > 7.0 or el / 10 > 5.0:
                print("健康检查不通过——停（K=15 硬件不可行分支）")
                sys.exit(1)
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

print("\n===== K=15 迭代预热轨迹 =====")
for r, m in enumerate(max_h, start=1):
    print(f"round {r}: curve max={m:.4f} (ctrl {ctrl_h[r-1]:.4f} / "
          f"seg {seg_h[r-1]:.4f})")
# 末段（连向 RI 的最后 2 段）与前段的分层结构
final_part = np.load(os.path.join(
    OUT, f"mc_ext_eps10_iter_k15_part{len(max_h)}.npz"))
seg_last = final_part["segmid_losses"]
print(f"末轮段中点分布: {np.round(seg_last, 3).tolist()}")
front = seg_last[:len(seg_last) // 2]
back = seg_last[len(seg_last) // 2:]
print(f"前半段 max={front.max():.4f}  后半段 max={back.max():.4f}")
if back.max() < 0.1:
    verdict = ("擦台消失入盆地：K=7 末段擦台是折线分辨率伪影——"
               "P3 链条需改写为'分辨率受限'")
elif back.max() >= 0.3:
    verdict = ("末段仍擦台：加密折线不消除——末段插值不可压缩是几何固有，"
               "P3 链条收口（注意 batch10 口径，绝对值与 P3c 不可直比）")
else:
    verdict = "末段降到中间带（0.1~0.3）：部分缓解，需看具体分布"
print(f"P3c-2 VERDICT: rounds max={np.round(max_h,4).tolist()} "
      f"末轮后半段 max={back.max():.4f}\n  -> {verdict}")

np.savez_compressed(
    os.path.join(OUT, "mc_ext_eps10_iter_k15.npz"),
    ctrl_maxes=np.array(ctrl_h), seg_maxes=np.array(seg_h),
    curve_maxes=np.array(max_h),
    meta=np.array(
        f"P3c-2: K=15 iterated warmup, eps=10% x {N_ROUNDS} rounds, "
        "each 150 steps fresh Adam, norm anchored at original RI; "
        "batch 10 (K=15 VRAM constraint) - absolute losses NOT "
        "comparable to P3c batch 50, compare structure only. "
        "Question: does denser polyline remove end-segment grazing?",
        dtype=np.str_))
print("saved mc_ext_eps10_iter_k15.npz")
print("DONE P3c-2 K15 iter warmup")
