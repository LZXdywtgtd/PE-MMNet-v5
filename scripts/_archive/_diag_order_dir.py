# -*- coding: utf-8 -*-
import sys; sys.path.insert(0, ".")
import numpy as np, torch
from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import resample_kpts_arclength

CKPT = "logs/training_history/v5a6_dprime_ordered/checkpoints/latest.pt"
model = create_v5_model("resnet18", image_channels=3, image_size=128, pretrained_2d=False, use_gp=True, min_kpts=8, max_kpts=16)
load_checkpoint(CKPT, model, map_location="cuda")
model.eval().cuda()
d = np.load("logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz")
sel = np.arange(10)
x2 = torch.from_numpy(d["x_2d"][sel]).float().cuda()
x1 = torch.from_numpy(d["x_1d"][sel]).float().cuda()
with torch.no_grad():
    out = model(x1, x2)
preds = out["keypoints"].cpu().numpy()
val = out["validity"].cpu().numpy()
print("validity mean=%.3f std=%.3f (trained if far from 0.479/0.247)" % (val.mean(), val.std()))

fwd_l, rev_l = [], []
for j, i in enumerate(sel):
    tk = torch.from_numpy(d["true_keypoints"][i]).float().cuda().unsqueeze(0)
    km = torch.from_numpy(d["keypoint_mask"][i]).bool().cuda().unsqueeze(0)
    tgt = resample_kpts_arclength(tk, km, 16)[0].cpu().numpy()
    pk = preds[j]
    d_fwd = np.abs(pk - tgt).sum()
    d_rev = np.abs(pk - tgt[::-1]).sum()
    fwd_l.append(d_fwd); rev_l.append(d_rev)
    if j < 10:
        print("s%d fwd=%.3f rev=%.3f  %s" % (i, d_fwd, d_rev, "REV-better" if d_rev < d_fwd else "FWD-better"))
fwd_l = np.array(fwd_l); rev_l = np.array(rev_l)
print("mean fwd=%.3f rev=%.3f" % (fwd_l.mean(), rev_l.mean()))
print("min-per-sample=%.3f (if much lower than both means -> direction flapping)" % (np.minimum(fwd_l, rev_l).mean()))
