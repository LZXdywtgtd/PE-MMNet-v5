# -*- coding: utf-8 -*-
"""选项 C 最后一问：failed 率是否依赖裂纹所在边（数据先验 vs 逐样本彩票）。

若某条边 failed 率显著更高 → 位置先验可学归因（数据侧重平衡可解）；
若四边接近 → 无数据侧判别量，位置锁定是训练动力学/架构问题。
口径：train（n=1000，统计力大）+ val，GT 质心最近边四分类。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch

from _diag_variance79 import CKPT_BASE, PX, POS_OK, FAIL_TH
from models import create_v5_model
from training.trainer_v5 import load_checkpoint

TRAIN_CACHE = "logs/sim_cache/sim_cache_p128_n1000_s424242_t6eb66e1b_th60.0_dp.npz"


def edge_of(gy, gx):
    """质心到四边距离的最小者：0=上 1=下 2=左 3=右。"""
    d = (gy, 127 - gy, gx, 127 - gx)
    return int(np.argmin(d))


def run(cache_path, tag):
    d = np.load(cache_path)
    px_mask = d["pixel_mask"]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]
    model = create_v5_model("resnet18", image_channels=3, image_size=128,
                            pretrained_2d=False, use_gp=True, min_kpts=8,
                            max_kpts=16, spatial_head=False)
    load_checkpoint(CKPT_BASE, model, map_location="cuda")
    model.eval().cuda()
    preds = []
    with torch.no_grad():
        for i in range(0, px_mask.shape[0], 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            preds.append(model(x1, x2)["keypoints"].cpu().numpy())
    preds = np.concatenate(preds)

    names = ("上", "下", "左", "右")
    cnt = np.zeros((4, 3), dtype=int)  # learned/mid/failed
    for i in pos_idx:
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        if px.shape[0] < 2:
            continue
        kt = torch.from_numpy(preds[i]).float().cuda()
        pt = torch.from_numpy(px).float().cuda()
        med = float(np.median(
            torch.cdist(kt, pt).min(dim=-1).values.cpu().numpy()) * PX)
        grp = 2 if med > FAIL_TH else (0 if med < POS_OK else 1)
        gc = px.mean(axis=0) * PX
        cnt[edge_of(gc[0], gc[1]), grp] += 1
    d.close()
    print(f"\n[{tag}] 按裂纹所在边的 failed 率")
    tot_l = tot_f = 0
    for e in range(4):
        n = cnt[e].sum()
        if n == 0:
            continue
        lr, fr = cnt[e, 0] / n * 100, cnt[e, 2] / n * 100
        tot_l += cnt[e, 0]; tot_f += cnt[e, 2]
        print(f"  {names[e]}: n={n:>4}  learned {lr:.0f}%  failed {fr:.0f}%")
    print(f"  合计: learned {tot_l/(tot_l+tot_f+cnt[:,1].sum())*100:.0f}% "
          f"failed {tot_f/(tot_l+tot_f+cnt[:,1].sum())*100:.0f}%")
    return cnt


def main():
    run(TRAIN_CACHE, "train n=1000")
    run("logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz",
        "val n=200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
