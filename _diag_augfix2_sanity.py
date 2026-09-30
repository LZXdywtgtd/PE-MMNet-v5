# -*- coding: utf-8 -*-
"""augfix2 快速复核：原 failed 代表样本 idx=13/176 的落点 + 通过率细节。"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import functools
print = functools.partial(print, flush=True)
import numpy as np
import torch

from _diag_variance79 import CKPT_BASE, VAL_CACHE, PX
from _diag_v5b_direction2 import UPPER
from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from _diag_v5b_direction2 import cham_of

CKPT = "logs/training_history/v5a6_dprime_augfix2/checkpoints/best.pt"

m = create_v5_model("resnet18", image_channels=3, image_size=128,
                    pretrained_2d=False, use_gp=True, min_kpts=8,
                    max_kpts=16, spatial_head=False)
st = load_checkpoint(CKPT, m, map_location="cuda")
print(f"best.pt epoch={st.get('epoch')} best_val_cov={st.get('best_val_coverage')}")
m.eval().cuda()
d = np.load(VAL_CACHE)
preds = []
with torch.no_grad():
    for i in range(0, d["x_2d"].shape[0], 50):
        x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
        x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
        preds.append(m(x1, x2)["keypoints"].cpu().numpy())
preds = np.concatenate(preds)

for i in (13, 176):
    px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
    gc = px.mean(axis=0) * PX
    pc = preds[i].mean(axis=0) * PX
    ch = cham_of(preds[i].astype(np.float64), px)
    print(f"idx={i}: GT质心(y,x)=({gc[0]:.0f},{gc[1]:.0f})  "
          f"pred质心=({pc[0]:.0f},{pc[1]:.0f})  chamfer={ch:.4f} ({ch*PX:.1f}px)")

chams = []
for i in range(d["x_2d"].shape[0]):
    px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
    if px.shape[0] < 2:
        continue
    chams.append(cham_of(preds[i].astype(np.float64), px))
chams = np.array(chams)
print(f"\nbest.pt: p50={np.median(chams):.4f} ratio={np.median(chams)/UPPER:.2f}  "
      f"通过 {float((chams <= UPPER*1.5).mean())*100:.0f}%  "
      f"worst10样本 chamfer: {np.sort(chams)[-10:].round(3)}")
d.close()
