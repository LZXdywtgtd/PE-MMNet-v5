# -*- coding: utf-8 -*-
"""判据 3/4 严格版验证（训练完成后跑）。

判据 3（跨输入方差）：pred_kpts 对不同输入的响应差异 ≥ 1e-3
判据 4（chamfer ≤ 上界×1.5）：pred_kpts→真像素 chamfer 不劣于
    "真关键点当预测"的 chamfer 上界的 1.5 倍

用法：
    python _eval_criteria34.py [ckpt_path]
缺省 checkpoint：logs/training_history/v5a6_dprime_resnet18/checkpoints/best.pt
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools

print = functools.partial(print, flush=True)

import numpy as np
import torch

from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import coverage_loss, catmull_rom_spline_torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/training_history/v5a6_dprime_resnet18/checkpoints/best.pt"
VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz"


def main():
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    print(f"ckpt epoch: {ckpt.get('epoch')}")
    model = create_v5_model(
        "resnet18", image_channels=3, image_size=128,
        pretrained_2d=ckpt.get("config", {}).get("pretrained_2d", False),
        use_gp=True, min_kpts=8, max_kpts=16,
    )
    load_checkpoint(CKPT, model, map_location="cuda")
    model.eval().cuda()

    d = np.load(VAL_CACHE)
    n_val = d["x_2d"].shape[0]
    px_mask = d["pixel_mask"]
    # 判据 3：全 200 个 val 样本 pred_kpts 跨输入方差
    # 判据 4：全部正样本 chamfer vs 真kpts 上界
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]
    print(f"val 正样本数: {len(pos_idx)}/{n_val}")

    B = 50
    preds, bboxes = [], []
    with torch.no_grad():
        for i in range(0, n_val, B):
            x2 = torch.from_numpy(d["x_2d"][i:i+B]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+B]).float().cuda()
            out = model(x1, x2)
            preds.append(out["keypoints"].cpu().numpy())
            bboxes.append(out["bbox"].cpu().numpy())
    preds = np.concatenate(preds)       # (200, 16, 2)
    bboxes = np.concatenate(bboxes)     # (200, 4)

    inter_var = preds.var(axis=0).mean()
    print(f"\n[判据3] 跨输入方差 inter_var = {inter_var:.6f}  (阈值 >= 1e-3)")
    c3 = inter_var >= 1e-3

    # chamfer（逐样本，双向对齐训练口径）
    from training.ordered_kp_loss import coverage_loss

    chamfers, uppers = [], []
    with torch.no_grad():
        for i in pos_idx:
            px = torch.from_numpy(
                d["true_crack_pixels"][i][d["pixel_mask"][i]]).float().cuda()
            if px.shape[0] < 2:
                continue
            pk = torch.from_numpy(preds[i]).float().cuda().unsqueeze(0)
            tk = torch.from_numpy(
                d["true_keypoints"][i][d["keypoint_mask"][i]]).float().cuda().unsqueeze(0)
            px = px.unsqueeze(0)
            c_pred = coverage_loss(pk, px, M=200, mode="chamfer").item()
            c_true = coverage_loss(tk, px, M=200, mode="chamfer").item()
            chamfers.append(c_pred)
            uppers.append(c_true)

    chamfers = np.array(chamfers)
    uppers = np.array(uppers)
    med_c = float(np.median(chamfers))
    med_u = float(np.median(uppers))
    ratio = med_c / max(med_u, 1e-9)
    print(f"[判据4] chamfer(pred) p50={med_c:.4f}  上界(true kpts) p50={med_u:.4f}"
          f"  ratio={ratio:.2f}  (阈值 <= 1.5)")
    c4 = ratio <= 1.5
    # 补充：bbox 跨样本活性 + 样条出界率
    bbox_std = bboxes.std(axis=0).mean()
    spline = catmull_rom_spline_torch(torch.from_numpy(preds).float().cuda(), 200)
    oob = float((spline.min(dim=-1).values < -0.05).any(dim=-1).float().mean() +
                (spline.max(dim=-1).values > 1.05).any(dim=-1).float().mean())
    print(f"[附] bbox 跨样本 std={bbox_std:.4f}  样条出界率={oob:.3f}")

    print("\n=== 判据汇总 ===")
    print(f"判据3 跨输入方差: {'PASS' if c3 else 'FAIL'}")
    print(f"判据4 chamfer 比: {'PASS' if c4 else 'FAIL'}")
    d.close()
    return 0 if (c3 and c4) else 1


if __name__ == "__main__":
    sys.exit(main())
