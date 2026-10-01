# -*- coding: utf-8 -*-
"""阶段 4 闸门：5-seed 通过率分布表（干净 GT 缓存口径）。

基线 seed 424242 = v5b_s3_gtfix_dp（--train_seed 缺省=seed）。
新 seed 424243-424246 = v5b_s4_seed*（--train_seed 分离，数据同 424242）。
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

VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz"
PX = 128
REQ = UPPER * 1.5

RUNS = [
    ("424242(base)", "logs/training_history/v5b_s3_gtfix_dp"),
    ("424243", "logs/training_history/v5b_s4_seed424243"),
    ("424244", "logs/training_history/v5b_s4_seed424244"),
    ("424245", "logs/training_history/v5b_s4_seed424245"),
    ("424246", "logs/training_history/v5b_s4_seed424246"),
]

d = np.load(VAL_CACHE)
n = d["x_2d"].shape[0]
pxs = [d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
       for i in range(n)]

rows = []
for tag, run in RUNS:
    ck = os.path.join(run, "checkpoints", "best.pt")
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
    chams = np.array([
        cham_of(preds[i].astype(np.float64), pxs[i]) * PX
        for i in range(n) if pxs[i].shape[0] >= 2
    ])
    npos = len(chams)
    passed = int((chams <= REQ * PX).sum())
    rows.append(dict(
        seed=tag, ep=st.get("epoch"), npos=npos, passed=passed,
        rate=passed / npos * 100,
        p50=float(np.median(chams)), p90=float(np.percentile(chams, 90)),
        mx=float(chams.max()), fail60=int((chams > 60).sum()),
    ))
    print(f"seed {tag}: ep{st.get('epoch')} 通过 {passed}/{npos} "
          f"({passed/npos*100:.1f}%)  p50={np.median(chams):.2f}px "
          f"p90={np.percentile(chams,90):.2f}  max={chams.max():.2f}  "
          f"failed60={int((chams>60).sum())}")
d.close()

print()
rates = np.array([r["rate"] for r in rows])
p50s = np.array([r["p50"] for r in rows])
print("===== 5-seed 分布表 =====")
print(f"通过率: min={rates.min():.1f}% max={rates.max():.1f}% "
      f"mean={rates.mean():.1f}% ± {rates.std():.2f}")
print(f"chamfer p50: min={p50s.min():.2f} max={p50s.max():.2f} "
      f"mean={p50s.mean():.2f} ± {p50s.std():.2f} px")
ok = all(r["rate"] >= 99.0 and r["fail60"] == 0 for r in rows)
print(f"闸门（全 seed ≥99% 且 failed60=0）：{'PASS' if ok else 'FAIL'}")
