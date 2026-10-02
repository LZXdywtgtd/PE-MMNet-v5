# -*- coding: utf-8 -*-
"""Round12 补强 P1：全量 197 正样本验证（50 样本口径的脆弱点 2）。

子任务串行（每个完成立即落盘 part npz，断点续跑基线）：
  0. AB  K=7 mc（opt_seed 4242）
  1. AC  K=7 mc（4242）
  2. ARI K=7 mc（4242）
  3. AB  K=7 三 opt_seed（42/137/2024）
  4. AC  z 投影消融（重优化 + 投影回平面双评）

显存：197 样本 × 5 中间点一次 backward 约需 15GB+（8GB 卡不可行），
改 micro-batch（50/块）梯度累积——loss=Σ(n_b/N)·loss_b 数学上与全
batch 一次 backward 等价（梯度线性）。50 样本版口径不变另存。

判据：全量 max 与 50 样本同量级（±20%）、PASS/FAIL 不翻转。
落盘 full197/f197_all.npz（+_part{ji}.npz 中间产物）。
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
                   "round12_mode_connectivity", "full197")
os.makedirs(OUT, exist_ok=True)
VAL_CACHE = os.path.join(PROJECT_ROOT, "logs", "sim_cache",
                         "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
STEPS = 150
LR = 1e-3
BZ = 50
print(f"device = {DEV}  full197 micro-batch = {BZ}")


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

# 全量正样本（与 landscape 抽样同一 pos_all 口径；50 版本是其子集）
d = np.load(VAL_CACHE)
pos_all = np.where(d["pixel_mask"].sum(axis=1) > 0)[0]
idx_all = pos_all
N_TOTAL = len(idx_all)
blocks = []
for s in range(0, N_TOTAL, BZ):
    ii = idx_all[s:s + BZ]
    blocks.append(dict(
        x1=torch.from_numpy(d["x_1d"][ii]).float().to(DEV),
        x2=torch.from_numpy(d["x_2d"][ii]).float().to(DEV),
        pk=torch.from_numpy(d["true_crack_pixels"][ii]).float().to(DEV),
        pm=torch.from_numpy(d["pixel_mask"][ii]).to(DEV),
        tk=torch.from_numpy(d["true_keypoints"][ii]).float().to(DEV),
        km=torch.from_numpy(d["keypoint_mask"][ii]).to(DEV),
        tb=torch.from_numpy(d["true_bbox"][ii]).float().to(DEV),
        w=len(ii) / N_TOTAL))
print(f"val 正样本 {len(pos_all)} 全量（{len(blocks)} 个 micro-batch）")

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


def _loss_on_block(out, blk):
    return loss_fn(
        pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
        true_bbox=blk["tb"], true_crack_pixels=blk["pk"],
        gp_module=None, pixel_mask=blk["pm"], true_kpts=blk["tk"],
        kpt_mask=blk["km"], pred_validity=out.get("validity"),
        true_position_6d=None,
        pred_position_6d=out.get("position_6d"))["total"]


@torch.no_grad()
def eval_sd(params):
    """全量平均 loss（分块加权）。"""
    model.load_state_dict(
        {k: (v.detach() if v.requires_grad else v) for k, v in params.items()},
        strict=True)
    model.eval()
    total = 0.0
    for blk in blocks:
        out = model(blk["x1"], blk["x2"])
        total += blk["w"] * _loss_on_block(out, blk).item()
    return total


def get_pair(pair, ri_seed=0):
    sd_A = load_sd("v5b_s4_seed424243")
    if pair == "AB":
        return sd_A, load_sd("v5b_s3_gtfix_dp"), "dp"
    if pair == "AC":
        return sd_A, load_sd("v5b_s4_seed424245"), "seed424245"
    return sd_A, random_init_sd(ri_seed), f"随机初始化(seed{ri_seed})"


def flat_np(sd):
    return np.concatenate(
        [(v.detach().cpu().numpy() if torch.is_tensor(v)
          else np.asarray(v)).astype(np.float64).ravel()
         for v in (sd[k] for k in FLOAT_KEYS)])


# 统一正则平面（与主实验一致）：center=dp, d1/d2=s424243/s424245
theta_c_np = flat_np(load_sd("v5b_s3_gtfix_dp"))
M = np.stack([flat_np(load_sd("v5b_s4_seed424243")) - theta_c_np,
              flat_np(load_sd("v5b_s4_seed424245")) - theta_c_np])
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


def run_mc(pair, K, opt_seed, tag, ri_seed=0):
    """全量 197 折线优化（micro-batch 梯度累积）。"""
    torch.manual_seed(opt_seed)
    np.random.seed(opt_seed % (2**31))
    sd_A, sd_B, label_B = get_pair(pair, ri_seed)
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

    def curve_loss_grad():
        """分块加权 loss 的梯度累积到 mid_params；返回标量用于打印。"""
        model.eval()
        total_val = 0.0
        for th in path_points()[1:-1]:
            for blk in blocks:
                out = functional_call(model, th, (blk["x1"], blk["x2"]))
                lo = _loss_on_block(out, blk)
                (blk["w"] * lo).backward()
                total_val += blk["w"] * lo.item()
        return total_val

    @torch.no_grad()
    def sample_path_losses(n_pts=100):
        ts = np.linspace(0.0, 1.0, n_pts)
        losses = np.full(n_pts, np.nan)
        pts = path_points()
        for j, t in enumerate(ts):
            x = t * (K - 1)
            seg = int(np.clip(np.floor(x), 0, K - 2))
            u = x - seg
            total = 0.0
            for blk in blocks:
                sd_mix = {}
                for k in FLOAT_KEYS:
                    sd_mix[k] = (pts[seg][k].detach() * (1 - u)
                                 + pts[seg + 1][k].detach() * u)
                for k, v in base_state.items():
                    if k not in sd_mix:
                        sd_mix[k] = v
                model.load_state_dict(sd_mix, strict=True)
                model.eval()
                out = model(blk["x1"], blk["x2"])
                total += blk["w"] * _loss_on_block(out, blk).item()
            losses[j] = total if np.isfinite(total) else np.nan
        return ts, losses

    ts_line, line_losses = sample_path_losses()

    mid_params = [th[k] for th in mids for k in PARAM_KEYS]
    opt = torch.optim.Adam(mid_params, lr=LR)
    hist_steps, hist_curves = [], []
    t_start = time.time()
    diverged = False
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss_val = curve_loss_grad()
        if not np.isfinite(loss_val):
            print(f"[{tag}] step {step}: NaN/Inf 发散，停")
            diverged = True
            break
        torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
        opt.step()
        if step % 5 == 0 or step == STEPS - 1:
            hist_steps.append(step)
            hist_curves.append(loss_val / (K - 2))
            if step % 50 == 0:
                el = time.time() - t_start
                eta = el / (step + 1) * (STEPS - step - 1)
                print(f"[{tag}] step {step:3d} "
                      f"mean_mid={loss_val/(K-2):.4f} "
                      f"({el/60:.1f}min, ETA {eta/60:.1f}min)")
    # 分阶段：优化后即落一次 live（防采样段崩溃丢优化结果）
    np.savez_compressed(
        os.path.join(OUT, f"f197_{tag}_live.npz"),
        hist_steps=np.array(hist_steps),
        hist_curves=np.array(hist_curves))

    n_ctrl = 2 * (K - 1)
    ts2, curve_losses = sample_path_losses(n_pts=n_ctrl)

    # 中间点 z 位移 + loss
    mid_z, mid_losses = [], []
    with torch.no_grad():
        for th in mids:
            vec = np.concatenate(
                [(th[k].detach().cpu().numpy().astype(np.float64)).ravel()
                 for k in FLOAT_KEYS])
            diff = vec - theta_c_np
            coef = diff @ M.T @ Ginv
            perp = diff - coef @ M
            mid_z.append(float(np.linalg.norm(perp)
                               / np.linalg.norm(theta_c_np)))
            total = 0.0
            for blk in blocks:
                out = functional_call(model, th, (blk["x1"], blk["x2"]))
                total += blk["w"] * _loss_on_block(out, blk).item()
            mid_losses.append(total)

    line_max = float(np.nanmax(line_losses))
    curve_max = float(np.nanmax(curve_losses))
    seg_jumps = np.abs(np.diff(curve_losses))
    print(f"[{tag}] 端点 A={lA:.4f} B({label_B})={lB:.4f} | "
          f"直线 max={line_max:.4f} 曲线 max={curve_max:.4f} | "
          f"max跳变={np.nanmax(seg_jumps):.4f}")
    print(f"[{tag}] 中间点 loss: "
          + " ".join(f"{v:.4f}" for v in mid_losses))
    print(f"[{tag}] 中间点 z: " + " ".join(f"{v:.4f}" for v in mid_z))
    return dict(pair=pair, K=K, opt_seed=opt_seed, label_B=label_B,
                lA=lA, lB=lB, diverged=diverged,
                ts_line=ts_line, line_losses=line_losses,
                ts_curve=ts2, curve_losses=curve_losses,
                hist_steps=np.array(hist_steps),
                hist_curves=np.array(hist_curves),
                mid_losses=np.array(mid_losses), mid_z=np.array(mid_z),
                line_max=line_max, curve_max=curve_max)


def run_ac_ablation(opt_seed=4242):
    """AC K=7 重优化 + z 投影回平面双评（全量口径）。"""
    torch.manual_seed(opt_seed)
    np.random.seed(opt_seed % (2**31))
    sd_A = load_sd("v5b_s4_seed424243")
    sd_B = load_sd("v5b_s4_seed424245")
    theta_A, theta_B = make_leaf(sd_A), make_leaf(sd_B)
    K = 7
    mids = [make_leaf(sd_A) for _ in range(1, K - 1)]
    with torch.no_grad():
        for i, p in enumerate(mids, start=1):
            t = i / (K - 1)
            for k in FLOAT_KEYS:
                p[k].copy_(theta_A[k].detach() * (1 - t)
                           + theta_B[k].detach() * t)

    def curve_loss_grad():
        model.eval()
        total_val = 0.0
        for th in mids:
            for blk in blocks:
                out = functional_call(model, th, (blk["x1"], blk["x2"]))
                lo = _loss_on_block(out, blk)
                (blk["w"] * lo).backward()
                total_val += blk["w"] * lo.item()
        return total_val

    mid_params = [th[k] for th in mids for k in PARAM_KEYS]
    opt = torch.optim.Adam(mid_params, lr=LR)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss_val = curve_loss_grad()
        if not np.isfinite(loss_val):
            print(f"step {step}: 发散，停")
            break
        torch.nn.utils.clip_grad_norm_(mid_params, 5.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step:3d} mean_mid={loss_val/(K-2):.4f}")

    l_opt, l_proj, z_frac = [], [], []
    with torch.no_grad():
        for th in mids:
            vec = np.concatenate(
                [(th[k].detach().cpu().numpy().astype(np.float64)).ravel()
                 for k in FLOAT_KEYS])
            l_opt.append(eval_sd(unflatten(vec)))
            diff = vec - theta_c_np
            coef = diff @ M.T @ Ginv
            vec_proj = theta_c_np + coef @ M
            l_proj.append(eval_sd(unflatten(vec_proj)))
            z_frac.append(float(np.linalg.norm(vec - vec_proj)
                                / np.linalg.norm(theta_c_np)))
    l_opt, l_proj, z_frac = map(np.array, (l_opt, l_proj, z_frac))
    for i in range(len(mids)):
        print(f"mid{i+1}: 优化后={l_opt[i]:.4f}  投回平面={l_proj[i]:.4f}  "
              f"z={z_frac[i]*100:.2f}%")
    return dict(l_opt=l_opt, l_proj=l_proj, z_frac=z_frac)


jobs = [("mc", ("AB", 7, 4242, "AB-K7")),
        ("mc", ("AC", 7, 4242, "AC-K7")),
        ("mc", ("ARI", 7, 4242, "ARI-K7")),
        ("mc", ("AB", 7, 42, "AB-s42")),
        ("mc", ("AB", 7, 137, "AB-s137")),
        ("mc", ("AB", 7, 2024, "AB-s2024")),
        ("abl", ("AC",))]

results = []
abl_result = None
for ji, (kind, args) in enumerate(jobs):
    t0 = time.time()
    if kind == "mc":
        r = run_mc(*args)
        results.append(r)
        payload = {f"r{i}_{k}": v for i, rr in enumerate(results)
                   for k, v in rr.items() if k not in ("label_B",)}
    else:
        abl_result = run_ac_ablation()
        payload = {f"r{i}_{k}": v for i, rr in enumerate(results)
                   for k, v in rr.items() if k not in ("label_B",)}
        payload.update({f"abl_{k}": v for k, v in abl_result.items()})
    part_path = os.path.join(OUT, f"f197_part{ji}.npz")
    np.savez_compressed(
        part_path,
        meta=np.array(
            f"P1 full197 PARTIAL {ji+1}/{len(jobs)} "
            f"({time.strftime('%H:%M:%S')}); all {N_TOTAL} pos samples, "
            f"micro-batch {BZ} grad accum (equiv full batch); "
            "Adam lr=1e-3 S=150 clip 5.0", dtype=np.str_),
        **payload)
    print(f">>> 阶段落盘 {ji+1}/{len(jobs)} -> {os.path.basename(part_path)} "
          f"({(time.time()-t0)/60:.1f}min)")

np.savez_compressed(
    os.path.join(OUT, "f197_all.npz"),
    meta=np.array(
        f"P1 full197 FINAL ({time.strftime('%H:%M:%S')}); all {N_TOTAL} "
        f"pos, micro-batch {BZ} grad accum; Adam lr=1e-3 S=150 clip 5.0; "
        "unified plane center=dp", dtype=np.str_),
    **{f"r{i}_{k}": v for i, rr in enumerate(results)
       for k, v in rr.items() if k not in ("label_B",)},
    **({f"abl_{k}": v for k, v in abl_result.items()} if abl_result else {}))
print("saved f197_all.npz")

verdicts = [f"{r['pair']}-K{r['K']}-s{r['opt_seed']}: max={r['curve_max']:.4f}"
            for r in results]
print("VERDICT " + " | ".join(verdicts))
print("DONE P1 full197")
