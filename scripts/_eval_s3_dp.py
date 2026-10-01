# -*- coding: utf-8 -*-
"""阶段 3 闸门：新 GT 缓存基线（v5b_s3_gtfix_dp）判据 3/4 评估。

预期：L 形污染清除后逐样本通过率 ≥99%（augfix2 在污染缓存上 93%）。
口径：val 197 正样本，判据 4 ratio = chamfer p50 / GT 下界 p50（0.0294），
阈值 1.5（chamfer ≤ 0.0441）；判据 3 跨输入方差 ≥1e-3。
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "lib"))
import functools
print = functools.partial(print, flush=True)
import numpy as np
import torch

from _diag_v5b_direction2 import cham_of, UPPER
from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import catmull_rom_spline_torch

CKPT = "logs/training_history/v5b_s3_gtfix_dp/checkpoints/best.pt"
CKPT_LATEST = "logs/training_history/v5b_s3_gtfix_dp/checkpoints/latest.pt"
VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz"
PX = 128
REQ = UPPER * 1.5

d = np.load(VAL_CACHE)
n = d["x_2d"].shape[0]

rows = []  # (cham_px, inter_var)
for ck in (CKPT, CKPT_LATEST):
    m = create_v5_model("resnet18", image_channels=3, image_size=128,
                        pretrained_2d=False, use_gp=True, min_kpts=8,
                        max_kpts=16, spatial_head=False)
    st = load_checkpoint(ck, m, map_location="cuda")
    m.eval().cuda()
    preds = []
    with torch.no_grad():
        for i in range(0, n, 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            preds.append(m(x1, x2)["keypoints"].cpu().numpy())
    preds = np.concatenate(preds)

    chams = []
    for i in range(n):
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        if px.shape[0] < 2:
            continue
        chams.append((i, cham_of(preds[i].astype(np.float64), px) * PX))
    arr = np.array([c for _, c in chams])
    pass_n = int((arr <= REQ * PX).sum())
    npos = len(arr)
    print(f"\n===== {os.path.basename(os.path.dirname(os.path.dirname(ck)))} "
          f"({os.path.basename(ck)}, ep{st.get('epoch')}) =====")
    print(f"正样本 {npos}  通过 {pass_n}（{pass_n/npos*100:.1f}%）  "
          f"未通过 {npos-pass_n}")
    print(f"chamfer px: p50={np.median(arr):.2f} p90={np.percentile(arr,90):.2f} "
          f"max={arr.max():.2f}  （阈 {REQ*PX:.2f}px）")
    print(f"ratio = {np.median(arr)/PX/UPPER:.3f}（阈 ≤1.5）  "
          f"failed(>60px)={int((arr>60).sum())}")
    bad = sorted(chams, key=lambda t: -t[1])[:8]
    print(f"最差 8: {[(i, round(c,1)) for i, c in bad]}")
    if ck == CKPT:
        # 判据 3：跨输入方差（同款口径：同 batch 重复输入 5 次的 kpt 方差）
        x2 = torch.from_numpy(d["x_2d"][:4]).float().cuda()
        x1 = torch.from_numpy(d["x_1d"][:4]).float().cuda()
        outs = []
        with torch.no_grad():
            for _ in range(5):
                outs.append(m(x1, x2)["keypoints"])
        inter_var = torch.stack(outs).var(dim=0).mean().item()
        print(f"判据 3 inter_var={inter_var:.4f}（阈 ≥1e-3）")
d.close()
