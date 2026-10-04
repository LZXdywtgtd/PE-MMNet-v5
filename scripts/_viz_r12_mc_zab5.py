# -*- coding: utf-8 -*-
"""Round12 过夜 P1 后续：z 投影消融补到 5 opt_seed（AB/AC 各 5）。

P0 单 seed（4242）已证 AC 消融与 AB 同构（z 0.59~0.76%、回弹
0.41~1.09）。本脚本把 AB/AC 各补到 5 个 opt_seed，检验
"0.7% 出平面位移 <-> barrier 量级回弹"的跨 run 稳定性。

判据（用户任务书）：z 位移落 0.5~1.0%、回弹 0.3~1.1 与单 seed
版一致；seed 4242 顺带复现原单次结果作内部一致性检查。

落盘：mc_ext_zab5_part_{PAIR}_{seed}.npz（每 run 独立，天然断点
续跑）+ mc_ext_zab5.npz（合并）。图 figH_zablation_5seed.png 由
_viz_r12_mc_extended_fig.py 的 zab5 模式渲染。
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

zz = np.load(os.path.join(LAND, "landscape_data.npz"), allow_pickle=False)
sample_idx = zz["sample_idx"]
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


def curve_loss(mids):
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


SD_A = load_sd("v5b_s4_seed424243")
END_B = {"AB": "v5b_s3_gtfix_dp", "AC": "v5b_s4_seed424245"}

PAIRS = ["AB", "AC"]
SEEDS = [4242, 42, 137, 2024, 7]
K = 7

results = {}
n_total = len(PAIRS) * len(SEEDS)
ji = 0
for PAIR in PAIRS:
    sd_B = load_sd(END_B[PAIR])
    theta_A_l, theta_B_l = make_leaf(SD_A), make_leaf(sd_B)
    for seed in SEEDS:
        ji += 1
        part_path = os.path.join(
            OUT, f"mc_ext_zab5_part_{PAIR}_{seed}.npz")
        if os.path.exists(part_path):
            dd = np.load(part_path)
            results[(PAIR, seed)] = {k: dd[k] for k in dd.files}
            print(f"[{PAIR}-s{seed}] part 已存在，跳过（断点续跑）")
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        mids = [make_leaf(SD_A) for _ in range(1, K - 1)]
        with torch.no_grad():
            for i, p in enumerate(mids, start=1):
                t = i / (K - 1)
                for k in FLOAT_KEYS:
                    p[k].copy_(theta_A_l[k].detach() * (1 - t)
                               + theta_B_l[k].detach() * t)
        mid_params = [th[k] for th in mids for k in PARAM_KEYS]
        opt = torch.optim.Adam(mid_params, lr=1e-3)
        for step in range(150):
            opt.zero_grad(set_to_none=True)
            loss = curve_loss(mids)
            if not torch.isfinite(loss):
                print(f"[{PAIR}-s{seed}] step {step}: 发散，停")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
            opt.step()
            if step % 50 == 0:
                el = (time.time() - t0) / 60
                print(f"[{PAIR}-s{seed}] step {step:3d} "
                      f"mean_mid={loss.item()/(K-2):.4f} ({el:.1f}min)")
        # 投影消融
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
        np.savez_compressed(part_path, l_opt=l_opt, l_proj=l_proj,
                            z_frac=z_frac)
        results[(PAIR, seed)] = dict(l_opt=l_opt, l_proj=l_proj,
                                     z_frac=z_frac)
        print(f"[{PAIR}-s{seed}] 回弹 {l_proj.min():.3f}~{l_proj.max():.3f} "
              f"z {z_frac.min()*100:.2f}~{z_frac.max()*100:.2f}% "
              f">>> 阶段产物 {ji}/{n_total} ({(time.time()-t0)/60:.1f}min)")

# ---------- 合并 + 判据 ----------
all_rows = []
for PAIR in PAIRS:
    for seed in SEEDS:
        r = results.get((PAIR, seed))
        if r is None:
            print(f"[{PAIR}-s{seed}] 缺失（run 失败），合并跳过该 run")
            continue
        all_rows.append((PAIR, seed, r["l_opt"], r["l_proj"],
                         r["z_frac"]))
        np.savez_compressed(
            os.path.join(OUT, "mc_ext_zab5.npz"),
            pairs=np.array([p for p, *_ in all_rows]),
            seeds=np.array([s for _, s, *_ in all_rows]),
            l_opt=np.stack([r[2] for r in all_rows]),
            l_proj=np.stack([r[3] for r in all_rows]),
            z_frac=np.stack([r[4] for r in all_rows]),
            meta=np.array(
                "5-seed z projection ablation, pairs AB/AC x opt_seeds "
                "[4242,42,137,2024,7], K=7 150steps; l_opt/l_proj/z_frac "
                "each shape (n_runs, K-2); plane center=dp d1=s424243 "
                "d2=s424245", dtype=np.str_))
print("\n===== 5-seed z 消融汇总 =====")
for PAIR, seed, lo, lp, zf in all_rows:
    print(f"{PAIR}-s{seed}: 回弹 {lp.min():.3f}~{lp.max():.3f} "
          f"(mean {lp.mean():.3f})  z {zf.min()*100:.2f}~"
          f"{zf.max()*100:.2f}% (mean {zf.mean()*100:.2f}%)")
proj_all = np.concatenate([r[3] for r in all_rows])
z_all = np.concatenate([r[4] for r in all_rows])
print(f"全部 {len(all_rows)} run 合计: 回弹范围 {proj_all.min():.3f}~"
      f"{proj_all.max():.3f}  z 范围 {z_all.min()*100:.2f}~"
      f"{z_all.max()*100:.2f}%")
n_ok = ((proj_all >= 0.3) & (proj_all <= 1.1)).mean()
z_ok = ((z_all >= 0.005) & (z_all <= 0.010)).mean()
print(f"判据命中: 回弹 0.3~1.1 -> {n_ok*100:.0f}%  "
      f"z 0.5~1.0% -> {z_ok*100:.0f}%")
# seed 4242 与原单次结果一致性检查
for PAIR in PAIRS:
    orig = f"mc_ext_zproj_ablation{'_ac' if PAIR == 'AC' else ''}.npz"
    op = os.path.join(OUT, orig)
    if (PAIR, 4242) in results and os.path.exists(op):
        od = np.load(op)
        r = results[(PAIR, 4242)]
        dz = float(np.abs(r["z_frac"] - od["z_frac"]).max())
        dl = float(np.abs(r["l_proj"] - od["l_proj"]).max())
        print(f"[一致性] {PAIR} s4242 vs 原{orig}: z 最大差 "
              f"{dz*100:.3f}pp  回弹最大差 {dl:.4f}")
saved = os.path.join(OUT, "mc_ext_zab5.npz")
print(f"saved {os.path.basename(saved)} ({len(all_rows)}/{n_total} runs)")
print("DONE zab5")
