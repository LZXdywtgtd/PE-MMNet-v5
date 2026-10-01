# -*- coding: utf-8 -*-
import sys; sys.path.insert(0, ".")
import numpy as np, torch
from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import resample_kpts_arclength, ordered_kpt_loss


VAL = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz"
d = np.load(VAL)
N = d["x_2d"].shape[0]
pm = d["pixel_mask"]
pos = np.where(pm.sum(axis=1) > 0)[0]

for tag in ["v1", "v2"]:
    ck = "logs/training_history/v5a6_dprime_ordered%s/checkpoints/latest.pt" % ("" if tag=="v1" else "_v2")
    model = create_v5_model("resnet18", image_channels=3, image_size=128, pretrained_2d=False, use_gp=True, min_kpts=8, max_kpts=16)
    load_checkpoint(ck, model, map_location="cuda")
    model.eval().cuda()
    preds = []
    B = 50
    with torch.no_grad():
        for i in range(0, N, B):
            x2 = torch.from_numpy(d["x_2d"][i:i+B]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+B]).float().cuda()
            preds.append(model(x1, x2)["keypoints"].cpu().numpy())
    preds = np.concatenate(preds)
    ord_l, mono = [], []
    for i in pos[:100]:
        tk = torch.from_numpy(d["true_keypoints"][i]).float().cuda().unsqueeze(0)
        km = torch.from_numpy(d["keypoint_mask"][i]).bool().cuda().unsqueeze(0)
        tgt = resample_kpts_arclength(tk, km, 16)
        pk = torch.from_numpy(preds[i]).float().cuda().unsqueeze(0)
        l = ordered_kpt_loss(pk, tk, km).item()
        ord_l.append(l)
        p = preds[i]
        g = tgt[0].cpu().numpy()
        d_fwd = np.abs(p - g).sum(axis=1).mean()
        d_rev = np.abs(p - g[::-1]).sum(axis=1).mean()
        mono.append(1.0 if d_rev < d_fwd else 0.0)
    print("%s: ordered_loss p50=%.3f  pred按反向对齐占比=%.0f%% (0%%=模型学会规范方向)" % (tag, np.median(ord_l), 100*np.mean(mono)))
print()
print("kd0.5 参照:")
model = create_v5_model("resnet18", image_channels=3, image_size=128, pretrained_2d=False, use_gp=True, min_kpts=8, max_kpts=16)
load_checkpoint("logs/training_history/v5a6_dprime_resnet18/checkpoints/latest.pt", model, map_location="cuda")
model.eval().cuda()
ord_l = []
with torch.no_grad():
    for i in pos[:100]:
        tk = torch.from_numpy(d["true_keypoints"][i]).float().cuda().unsqueeze(0)
        km = torch.from_numpy(d["keypoint_mask"][i]).bool().cuda().unsqueeze(0)
        x2 = torch.from_numpy(d["x_2d"][i:i+1]).float().cuda()
        x1 = torch.from_numpy(d["x_1d"][i:i+1]).float().cuda()
        pk = model(x1, x2)["keypoints"]
        ord_l.append(ordered_kpt_loss(pk, tk, km).item())
print("kd0.5: ordered_loss p50=%.3f (从未被有序监督训练的基线)" % np.median(ord_l))
