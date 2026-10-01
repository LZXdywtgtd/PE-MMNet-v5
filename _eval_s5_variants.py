# -*- coding: utf-8 -*-
"""阶段 5 闸门：5 变体对照表（干净 GT 缓存，同 seed 424242）。"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
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
    ("resnet18(base)", "resnet18", "logs/training_history/v5b_s3_gtfix_dp"),
    ("swin_yolo", "swin_yolo", "logs/training_history/v5b_s5_swin_yolo"),
    ("vit_yolo", "vit_yolo", "logs/training_history/v5b_s5_vit_yolo"),
    ("detr", "detr", "logs/training_history/v5b_s5_detr"),
    ("swin_yolo_patchtst", "swin_yolo_patchtst",
     "logs/training_history/v5b_s5_swin_patchtst"),
]

d = np.load(VAL_CACHE)
n = d["x_2d"].shape[0]
pxs = [d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
       for i in range(n)]
idxs = [i for i in range(n) if pxs[i].shape[0] >= 2]

results = []
for tag, variant, run in RUNS:
    ck = os.path.join(run, "checkpoints", "best.pt")
    # 变体类签名不含 spatial_head（仅 resnet18 类有），per-variant 传参
    kw = dict(image_channels=3, image_size=128, pretrained_2d=False,
              use_gp=True, min_kpts=8, max_kpts=16)
    if variant == "resnet18":
        kw["spatial_head"] = False
    m = create_v5_model(variant, **kw)
    st = load_checkpoint(ck, m, map_location="cuda")
    m.eval().cuda()
    preds = []
    with torch.no_grad():
        for i in range(0, n, 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            preds.append(m(x1, x2)["keypoints"].cpu().numpy())
    preds = np.concatenate(preds)
    chams = np.array([cham_of(preds[i].astype(np.float64), pxs[i]) * PX
                      for i in idxs])
    npos = len(chams)
    passed = int((chams <= REQ * PX).sum())
    results.append(dict(tag=tag, ep=st.get("epoch"), passed=passed,
                        npos=npos, rate=passed / npos * 100,
                        p50=float(np.median(chams)),
                        mx=float(chams.max()),
                        fail60=int((chams > 60).sum())))
    print(f"{tag:20s}: ep{st.get('epoch')} 通过 {passed}/{npos} "
          f"({passed/npos*100:.1f}%)  p50={np.median(chams):.2f}px "
          f"max={chams.max():.2f}  failed60={int((chams>60).sum())}")
d.close()

print()
print("===== 5 变体对照表 =====")
print(f"{'变体':22s} {'bestEp':>7s} {'通过率':>9s} {'p50(px)':>8s} "
      f"{'max(px)':>8s} {'failed60':>8s}")
for r in results:
    print(f"{r['tag']:22s} {r['ep']:>7} {r['passed']}/{r['npos']}"
          f"({r['rate']:>5.1f}%) {r['p50']:>8.2f} {r['mx']:>8.2f} "
          f"{r['fail60']:>8}")
ok = all(r["rate"] >= 99.0 and r["fail60"] == 0 for r in results)
print(f"闸门（全变体 ≥99% 且 failed60=0）：{'PASS' if ok else 'FAIL'}")
