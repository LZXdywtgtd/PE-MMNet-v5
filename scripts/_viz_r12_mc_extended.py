# -*- coding: utf-8 -*-
"""Round12 工作项 4 补充实验：mode connectivity 结论稳健性三验证。

实验 1（--exp z）：AB/AC 优化后控制点到 landscape 正则平面的垂直位移
    z_i = ||theta_i - (theta_i 在平面上投影)|| / ||theta_c||（相对中心范数）
    判据：max z > 10% = 真绕行；< 2% = 平面内伪影风险（停）。
实验 2（--exp k15）：K=15 三配对（AB/AC/ARI），与 K=7 对照分辨率敏感性。
    判据：ARI 若 PASS 停（FAIL 可能是 K 不足）；AB/AC max<0.03=K 不足。
实验 3（--exp seed3）：AB K=7 三 seed（42/137/2024）优化后 max 分布。
    判据：CV<10% 稳定；>20% 报分布；>30% 停。

口径与主实验一致：端点固定、OrderedKeypointLoss total、50 正样本
(default_rng(424242))、Adam lr=1e-3、S=150 步、grad clip 5.0、BN eval。
数据落盘 round12_mode_connectivity/mc_ext_{zdisplace,k15,3seed}.npz。
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
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import OrderedKeypointLoss

EXP = sys.argv[1] if len(sys.argv) > 1 else "z"
OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_mode_connectivity")
os.makedirs(OUT, exist_ok=True)
VAL_CACHE = os.path.join(PROJECT_ROOT, "logs", "sim_cache",
                         "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")
TH = os.path.join(PROJECT_ROOT, "logs", "training_history")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 150
LR = 1e-3
N_SAMPLES = 50
# k15ari：ARI 单配对 + 中间点 batch 降 10（显存门控版，
# 每 30 步落盘 partial——上次三配对一个 npz 被杀丢 2.2h 的修复）
BATCH_OVERRIDE = 10 if EXP == "k15ari" else N_SAMPLES
print(f"device = {DEV}  exp = {EXP}  batch = {BATCH_OVERRIDE}")


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
sample_idx_full = z["sample_idx"]
sample_idx = sample_idx_full[:BATCH_OVERRIDE]
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
    lo = loss_fn(
        pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
        true_bbox=tb, true_crack_pixels=pk,
        gp_module=getattr(model, "gp_module", None),
        pixel_mask=pm, true_kpts=tk, kpt_mask=km,
        pred_validity=out.get("validity"),
        true_position_6d=None,
        pred_position_6d=out.get("position_6d"))
    return lo


def get_pair(pair):
    sd_A = load_sd("v5b_s4_seed424243")
    if pair == "AB":
        return sd_A, load_sd("v5b_s3_gtfix_dp"), "dp"
    if pair == "AC":
        return sd_A, load_sd("v5b_s4_seed424245"), "seed424245"
    return sd_A, random_init_sd(0), "随机初始化"


# 正则平面（与主实验/landscape 统一）：center=dp, d1/d2=s424243/s424245
sd_center = load_sd("v5b_s3_gtfix_dp")


def flat_np(sd):
    return np.concatenate(
        [(v.detach().cpu().numpy() if torch.is_tensor(v)
          else np.asarray(v)).astype(np.float64).ravel()
         for v in (sd[k] for k in FLOAT_KEYS)])


theta_c_np = flat_np(sd_center)
M = np.stack([flat_np(load_sd("v5b_s4_seed424243")) - theta_c_np,
              flat_np(load_sd("v5b_s4_seed424245")) - theta_c_np])
Ginv = np.linalg.inv(M @ M.T)
# 平面正交基（12M 维里的 2 维子空间的补空间方向不唯一，但"到平面的
# 垂直距离"唯一——直接算 diff - proj 即可，无需显式正交基）


def z_displacement(sd):
    """到正则平面的垂直距离 / 中心权重范数。"""
    diff = flat_np(sd) - theta_c_np
    coef = diff @ M.T @ Ginv                # 面内坐标 (alpha1, alpha2)
    perp = diff - coef @ M                  # 垂直分量
    return float(np.linalg.norm(perp) / np.linalg.norm(theta_c_np))


def run_pair(pair, K, opt_seed, tag):
    """单次折线优化，返回曲线采样/优化史/中间点 z 位移/中间点 loss。"""
    torch.manual_seed(opt_seed)
    np.random.seed(opt_seed % (2**31))
    sd_A, sd_B, label_B = get_pair(pair)
    theta_A = make_leaf(sd_A)
    theta_B = make_leaf(sd_B)
    with torch.no_grad():
        lA = eval_sd(theta_A)["total"].item()
        lB = eval_sd(theta_B)["total"].item()
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
        pts = path_points()
        model.eval()
        total = 0.0
        for th in pts[1:-1]:
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
    part_path = os.path.join(OUT, f"mc_ext_{sfx}_live_{tag_}.npz")

    def save_live(step):
        np.savez_compressed(
            part_path,
            meta=np.array(
                f"LIVE {tag} step {step}/{STEPS} "
                f"({time.strftime('%H:%M:%S')}); pair={pair} K={K} "
                f"batch={BATCH_OVERRIDE}; hist so far", dtype=np.str_),
            hist_steps=np.array(hist_steps),
            hist_curves=np.array(hist_curves))
        print(f">>> live 落盘 step {step} -> {os.path.basename(part_path)}")

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
        if step % 5 == 0 or step == STEPS - 1:
            hist_steps.append(step)
            hist_curves.append(loss.item() / (K - 2))
            if step % 25 == 0:
                el = time.time() - t_start
                eta = el / (step + 1) * (STEPS - step - 1)
                print(f"[{tag}] step {step:3d} "
                      f"mean_mid={loss.item()/(K-2):.4f} "
                      f"({el/60:.1f}min, ETA {eta/60:.1f}min)")
        if EXP == "k15ari" and step in (10, 29) or \
                (EXP == "k15ari" and step % 30 == 29 and step > 29):
            save_live(step)
        if EXP == "k15ari" and step == 10:
            mem = torch.cuda.max_memory_allocated() / 2**30
            el = time.time() - t_start
            print(f"[健康检查] step10: 峰值显存 {mem:.2f} GiB, "
                  f"前10步均耗时 {el/10:.2f}s")
            if mem > 7.0 or el / 10 > 5.0:
                print("健康检查不通过（显存>7GiB 或 单步>5s）——停，"
                      "结论记'K=15 在此硬件不可行'")
                diverged = True
                break

    # 优化后曲线只采控制点 + 段中点（K-1 个折点 + K-1 个段中点，
    # 覆盖 max 所在的折点邻域；每 step 不采样，固定开销砍半以上）
    n_ctrl = 2 * (K - 1)
    ts2, curve_losses = sample_path_losses(n_pts=n_ctrl)

    mid_z = []
    mid_losses = []
    with torch.no_grad():
        for th in mids:
            sd_np = {k: th[k].detach().cpu().numpy().astype(np.float64)
                     for k in FLOAT_KEYS}
            mid_z.append(z_displacement(sd_np))
            out = functional_call(model, th, (x1, x2))
            lo = loss_fn(
                pred_bbox=out["bbox"], pred_kpts=out["keypoints"],
                true_bbox=tb, true_crack_pixels=pk,
                gp_module=None, pixel_mask=pm, true_kpts=tk, kpt_mask=km,
                pred_validity=out.get("validity"),
                true_position_6d=None,
                pred_position_6d=out.get("position_6d"))
            mid_losses.append(lo["total"].item())

    line_max = float(np.nanmax(line_losses))
    curve_max = float(np.nanmax(curve_losses))
    # 段平滑度：曲线相邻采样点跳变（过参数化钻空子会剧烈抖动）
    seg_jumps = np.abs(np.diff(curve_losses))
    print(f"[{tag}] 端点 A={lA:.4f} B({label_B})={lB:.4f} | "
          f"直线 max={line_max:.4f} 曲线 max={curve_max:.4f} "
          f"比值={curve_max/max(line_max,1e-9):.2f} | "
          f"max跳变={np.nanmax(seg_jumps):.4f} mean跳变={np.nanmean(seg_jumps):.4f}")
    print(f"[{tag}] 中间点 loss: " + " ".join(f"{v:.4f}" for v in mid_losses))
    print(f"[{tag}] 中间点 z 位移: " + " ".join(f"{v:.4f}" for v in mid_z))
    return dict(
        pair=pair, K=K, opt_seed=opt_seed, label_B=label_B,
        lA=lA, lB=lB, diverged=diverged,
        ts_line=ts_line, line_losses=line_losses,
        ts_curve=ts2, curve_losses=curve_losses,
        hist_steps=np.array(hist_steps), hist_curves=np.array(hist_curves),
        mid_losses=np.array(mid_losses), mid_z=np.array(mid_z),
        line_max=line_max, curve_max=curve_max)


sfx = {"z": "zdisplace", "k15": "k15", "k15ari": "k15ari",
       "seed3": "3seed"}[EXP]
results = []
final_path = os.path.join(OUT, f"mc_ext_{sfx}.npz")
jobs = {"z": [("AB", 7, 4242, "AB-K7"), ("AC", 7, 4242, "AC-K7")],
        "k15": [("AB", 15, 4242, "AB-K15"), ("AC", 15, 4242, "AC-K15"),
                ("ARI", 15, 4242, "ARI-K15")],
        "k15ari": [("ARI", 15, 4242, "ARI-K15-b10")],
        "seed3": [("AB", 7, 42, "AB-seed42"), ("AB", 7, 137, "AB-seed137"),
                  ("AB", 7, 2024, "AB-seed2024")]}[EXP]

# 分阶段落盘：每配对完成立即追加写入（断点续跑基线——上次 K=15
# 三配对一个 npz 设计导致被杀后 2.2 小时全丢，此为修复）
for ji, (pair, K_, s_, tag_) in enumerate(jobs):
    r = run_pair(pair, K=K_, opt_seed=s_, tag=tag_)
    results.append(r)
    part_path = os.path.join(OUT, f"mc_ext_{sfx}_part{ji}.npz")
    np.savez_compressed(
        part_path,
        meta=np.array(
            f"MC extended exp={EXP} PARTIAL {ji+1}/{len(jobs)} "
            f"({time.strftime('%H:%M:%S')}); unified plane center=dp; "
            "endpoints fixed; Adam lr=1e-3 S=150 clip 5.0; "
            "loss=OKL total 50 pos (default_rng(424242)); "
            "z_disp=||perp||/||theta_c||", dtype=np.str_),
        **{f"r{i}_{k}": v for i, rr in enumerate(results)
           for k, v in rr.items() if k not in ("label_B",)})
    print(f">>> 阶段落盘 {ji+1}/{len(jobs)} -> {os.path.basename(part_path)}")

np.savez_compressed(
    final_path,
    meta=np.array(
        f"MC extended exp={EXP} FINAL; unified plane center=dp "
        "(same as landscape); endpoints fixed; Adam lr=1e-3 S=150 "
        "clip 5.0; loss=OrderedKeypointLoss total 50 pos "
        "(default_rng(424242)); z_disp = ||perp||/||theta_center||",
        dtype=np.str_),
    **{f"r{i}_{k}": v for i, r in enumerate(results)
       for k, v in r.items() if k not in ("label_B",)})
print(f"saved {os.path.basename(final_path)}")
verdicts = []
for r in results:
    verdicts.append(f"{r['pair']}-K{r['K']}-s{r['opt_seed']}: "
                    f"max={r['curve_max']:.4f}")
print("VERDICT " + " | ".join(verdicts))
print(f"DONE MC 补充实验 {EXP}")
