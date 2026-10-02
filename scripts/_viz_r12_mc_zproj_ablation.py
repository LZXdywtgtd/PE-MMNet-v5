# -*- coding: utf-8 -*-
"""Round12 工作项 4 补充实验 1b：z 投影消融（AB/AC K=7，argv 选配对）。

实验 1 发现中间点 z 位移仅 0.57%~0.78%（<2% 判据），为区分
"优化伪影"与"出平面绕行"，把优化后中间点投影回正则平面
（z 分量清零）再评 loss：
- 若 loss 保持低位 -> 低位与 z 无关，z 位移是装饰
- 若 loss 回弹 barrier 量级 -> 0.7% 出平面位移是绕行的全部所在

用法：python _viz_r12_mc_zproj_ablation.py [AB|AC]（默认 AB）
复现口径：K=7、Adam lr=1e-3、150 步、opt_seed 4242（同实验 1）。
落盘 mc_ext_zproj_ablation{_ac,}.npz；本脚本只产数据，图由
_viz_r12_mc_extended_fig.py 统一渲染。
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
from training.ordered_kp_loss import OrderedKeypointLoss

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_mode_connectivity")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
LAND = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                    "round12_loss_landscape")
print(f"device = {DEV}")


def load_sd(run):
    ck = torch.load(os.path.join(TH, run, "checkpoints", "best.pt"),
                    map_location="cpu", weights_only=False)
    return ck.get("model_state_dict", ck)


def build_model():
    return create_v5_model("resnet18", image_channels=3, image_size=128,
                           pretrained_2d=False, use_gp=True, min_kpts=8,
                           max_kpts=16, spatial_head=False).to(DEV)


model = build_model()
base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
FLOAT_KEYS = [k for k, v in base_state.items() if v.dtype.is_floating_point]
PARAM_KEYS = set(dict(model.named_parameters()).keys())

z = np.load(os.path.join(LAND, "landscape_data.npz"), allow_pickle=False)
sample_idx = z["sample_idx"]
d = np.load(os.path.join(PROJECT_ROOT, "logs", "sim_cache",
            "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz"))
x1 = torch.from_numpy(d["x_1d"][sample_idx]).float().to(DEV)
x2 = torch.from_numpy(d["x_2d"][sample_idx]).float().to(DEV)
pk = torch.from_numpy(d["true_crack_pixels"][sample_idx]).float().to(DEV)
pm = torch.from_numpy(d["pixel_mask"][sample_idx]).to(DEV)
tk = torch.from_numpy(d["true_keypoints"][sample_idx]).float().to(DEV)
km = torch.from_numpy(d["keypoint_mask"][sample_idx]).to(DEV)
tb = torch.from_numpy(d["true_bbox"][sample_idx]).float().to(DEV)
loss_fn = OrderedKeypointLoss(
    lambda_coverage=1.0, lambda_bbox=1.0, lambda_gp=0.1, lambda_poisson=0.05,
    lambda_kpt_direct=0.5, lambda_ordered=0.0, lambda_validity=0.0,
    lambda_position_6d=0.0, lambda_uv_projection=0.0)


def flat_np(sd):
    return np.concatenate(
        [(v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
         .astype(np.float64).ravel() for v in (sd[k] for k in FLOAT_KEYS)])


theta_c = flat_np(load_sd("v5b_s3_gtfix_dp"))
M = np.stack([flat_np(load_sd("v5b_s4_seed424243")) - theta_c,
              flat_np(load_sd("v5b_s4_seed424245")) - theta_c])
Ginv = np.linalg.inv(M @ M.T)


def unflatten(vec):
    out, i = {}, 0
    for k in FLOAT_KEYS:
        n = base_state[k].numel()
        out[k] = torch.from_numpy(
            vec[i:i + n].reshape(base_state[k].shape)
            .astype(np.float32)).to(DEV)
        i += n
    for k, v in base_state.items():
        if k not in out:
            out[k] = v.to(DEV)
    return out


@torch.no_grad()
def eval_vec(vec):
    model.load_state_dict(unflatten(vec), strict=True)
    model.eval()
    out = model(x1, x2)
    lo = loss_fn(pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
                 true_bbox=tb, true_crack_pixels=pk, gp_module=None,
                 pixel_mask=pm, true_kpts=tk, kpt_mask=km,
                 pred_validity=out.get("validity"),
                 true_position_6d=None,
                 pred_position_6d=out.get("position_6d"))
    return lo["total"].item()


def make_leaf(sd_like):
    return {k: (v.detach().clone().to(DEV).requires_grad_(True)
                if k in PARAM_KEYS else v.detach().clone().to(DEV))
            for k, v in sd_like.items()}


# ---------- K=7 重优化（同实验 1，配对由 argv 选）----------
PAIR = sys.argv[1].upper() if len(sys.argv) > 1 else "AB"
assert PAIR in ("AB", "AC"), f"未知配对 {PAIR}"
torch.manual_seed(4242)
np.random.seed(4242)
sd_A = load_sd("v5b_s4_seed424243")
sd_B = load_sd("v5b_s3_gtfix_dp" if PAIR == "AB"
               else "v5b_s4_seed424245")
theta_A, theta_B = make_leaf(sd_A), make_leaf(sd_B)
K = 7
mids = [make_leaf(sd_A) for _ in range(1, K - 1)]
with torch.no_grad():
    for i, p in enumerate(mids, start=1):
        t = i / (K - 1)
        for k in FLOAT_KEYS:
            p[k].copy_(theta_A[k].detach() * (1 - t)
                       + theta_B[k].detach() * t)


def curve_loss():
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
    return total


mid_params = [th[k] for th in mids for k in PARAM_KEYS]
opt = torch.optim.Adam(mid_params, lr=1e-3)
for step in range(150):
    opt.zero_grad(set_to_none=True)
    loss = curve_loss()
    if not torch.isfinite(loss):
        print(f"step {step}: 发散，停")
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d} mean_mid={loss.item()/(K-2):.4f}")
print(f"{PAIR} K=7 重优化完成")

# ---------- 投影消融 ----------
l_opt, l_proj, z_frac = [], [], []
for th in mids:
    vec = flat_np({k: th[k] for k in FLOAT_KEYS})
    l_opt.append(eval_vec(vec))
    diff = vec - theta_c
    coef = diff @ M.T @ Ginv
    vec_proj = theta_c + coef @ M
    l_proj.append(eval_vec(vec_proj))
    z_frac.append(float(np.linalg.norm(vec - vec_proj)
                        / np.linalg.norm(theta_c)))
l_opt, l_proj, z_frac = map(np.array, (l_opt, l_proj, z_frac))
for i in range(len(mids)):
    print(f"mid{i+1}: 优化后={l_opt[i]:.4f}  投回平面={l_proj[i]:.4f}  "
          f"z={z_frac[i]*100:.2f}%")

np.savez_compressed(
    os.path.join(OUT, f"mc_ext_zproj_ablation{'_ac' if PAIR == 'AC' else ''}.npz"),
    l_opt=l_opt, l_proj=l_proj, z_frac=z_frac,
    meta=np.array(
        f"{PAIR} K=7 opt(4242) 150steps; per-mid: loss optimized vs loss "
        "after projecting z-component to zero (back to unified plane "
        "center=dp); z_frac=||perp||/||theta_c||", dtype=np.str_))
print(f"saved mc_ext_zproj_ablation{'_ac' if PAIR == 'AC' else ''}.npz")
print(f"DONE MC 补充实验 1b z 投影消融 {PAIR}")
