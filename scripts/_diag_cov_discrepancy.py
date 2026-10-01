# -*- coding: utf-8 -*-
"""裁决：训练 CSV 的 val_coverage=0.021 vs 离线实测 spline chamfer=0.578 差 27 倍。

同指标（catmull-rom 200 采样 + 双向 chamfer，归一化坐标）同 checkpoint
不可能差 27 倍。两路都复算：
  A. 训练同款 coverage_loss() 直接喂离线 pred + val 缓存像素
  B. cham_of（判据 4 口径）
另打 checkpoint 元数据（epoch/best）+ best.pt 同口径。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "lib"))

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch

from _diag_variance79 import CKPT_BASE, VAL_CACHE
from _diag_v5b_direction2 import cham_of, UPPER
from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import coverage_loss

CKPT_LATEST = "logs/training_history/v5a6_dprime_augfix/checkpoints/latest.pt"
CKPT_BEST = "logs/training_history/v5a6_dprime_augfix/checkpoints/best.pt"


def get_preds(ckpt):
    m = create_v5_model("resnet18", image_channels=3, image_size=128,
                        pretrained_2d=False, use_gp=True, min_kpts=8,
                        max_kpts=16, spatial_head=False)
    st = load_checkpoint(ckpt, m, map_location="cuda")
    print(f"  ckpt={os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}"
          f"  epoch={st.get('epoch')}  best_val_loss={st.get('best_val_loss')}"
          f"  best_val_cov={st.get('best_val_coverage')}")
    m.eval().cuda()
    d = np.load(VAL_CACHE)
    preds = []
    with torch.no_grad():
        for i in range(0, d["x_2d"].shape[0], 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            preds.append(m(x1, x2)["keypoints"].cpu().numpy())
    preds = np.concatenate(preds)
    return d, preds


def main():
    for ckpt in (CKPT_LATEST, CKPT_BEST, CKPT_BASE):
        print(f"\n===== {ckpt} =====")
        d, preds = get_preds(ckpt)

        covs_train_style, covs_chamof = [], []
        for i in range(d["x_2d"].shape[0]):
            px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
            if px.shape[0] < 2:
                continue
            pk = torch.from_numpy(preds[i]).float().cuda().unsqueeze(0)
            pix = torch.from_numpy(px).float().cuda().unsqueeze(0)
            mask = torch.ones(1, px.shape[0], dtype=torch.bool).cuda()
            covs_train_style.append(float(coverage_loss(pk, pix, M=200,
                                                        mode="chamfer",
                                                        pixel_mask=mask)))
            covs_chamof.append(cham_of(preds[i].astype(np.float64), px))

        a = np.array(covs_train_style)
        b = np.array(covs_chamof)
        print(f"  n={len(a)}")
        print(f"  A. coverage_loss(训练同款): p50={np.median(a):.4f}  "
              f"mean={a.mean():.4f}  min={a.min():.4f}  max={a.max():.4f}")
        print(f"  B. cham_of(判据4口径):     p50={np.median(b):.4f}  "
              f"mean={b.mean():.4f}  min={b.min():.4f}  max={b.max():.4f}")
        print(f"  相关系数 A~B: {np.corrcoef(a, b)[0,1]:.3f}")
        print(f"  判据4 ratio(chamof p50/0.0294): {np.median(b)/UPPER:.2f}")
        d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
