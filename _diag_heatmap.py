# -*- coding: utf-8 -*-
"""spatial head 中间诊断：heatmap 头是否已定位裂纹（20ep 快照）。

判读：若 pred heatmap 的 top-k 单元对 GT 正单元有召回（AUC 明显>0.5），
说明空间通路在学"裂纹在哪"，续训有意义；若 AUC≈0.5（随机），
空间通路本身没学到，20ep 失败更可能是监督/结构问题。
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
from models.spatial_kpt_head import gt_heatmap_target

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/training_history/v5a6_dprime_spatial/checkpoints/latest.pt"
VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz"


def main():
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    state_keys = ckpt.get("model_state_dict", {}).keys()
    has_spatial = any(k.startswith("spatial_head.") for k in state_keys)
    model = create_v5_model(
        "resnet18", image_channels=3, image_size=128,
        pretrained_2d=False, use_gp=True, min_kpts=8, max_kpts=16,
        spatial_head=has_spatial,
    )
    load_checkpoint(CKPT, model, map_location="cuda")
    model.eval().cuda()

    d = np.load(VAL_CACHE)
    px_mask = d["pixel_mask"]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]

    aucs, prec_at_5, prec_at_10 = [], [], []
    B = 50
    with torch.no_grad():
        for i0 in range(0, len(pos_idx), B):
            sel = pos_idx[i0:i0 + B]
            x2 = torch.from_numpy(d["x_2d"][sel]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][sel]).float().cuda()
            out = model(x1, x2)
            hm = out["heatmap"]  # (B,1,8,8)
            pm = torch.from_numpy(px_mask[sel]).to(x2.device)
            tp = torch.from_numpy(d["true_crack_pixels"][sel]).float().to(x2.device)
            tgt = gt_heatmap_target(pm, tp, hm.size(2), hm.size(3))
            hm_f = hm.flatten(1)
            tg_f = tgt.flatten(1)
            for b in range(hm_f.size(0)):
                p = hm_f[b]
                t = tg_f[b]
                n_pos = int(t.sum().item())
                if n_pos == 0:
                    continue
                # rank AUC: P(pred_pos > pred_neg)
                pos_vals = p[t > 0.5]
                neg_vals = p[t <= 0.5]
                gt_pos = (pos_vals.unsqueeze(1) > neg_vals.unsqueeze(0)).float().mean().item()
                ties = (pos_vals.unsqueeze(1) == neg_vals.unsqueeze(0)).float().mean().item()
                aucs.append(gt_pos + 0.5 * ties)
                # precision@k：前 k 个预测单元中 GT 正单元占比
                order = torch.argsort(p, descending=True)
                for k, acc in ((5, prec_at_5), (10, acc if False else prec_at_10)):
                    topk = t[order[:k]]
                    acc.append(topk.sum().item() / min(k, n_pos))

    print(f"样本数: {len(aucs)}")
    print(f"heatmap rank-AUC: mean={np.mean(aucs):.3f}  median={np.median(aucs):.3f}"
          f"  (随机=0.5, 完美=1.0)")
    print(f"precision@5:  mean={np.mean(prec_at_5):.3f}")
    print(f"precision@10: mean={np.mean(prec_at_10):.3f}")
    hi = sum(1 for a in aucs if a >= 0.8)
    lo = sum(1 for a in aucs if a <= 0.6)
    print(f"AUC>=0.8 样本: {hi}/{len(aucs)}   AUC<=0.6 样本: {lo}/{len(aucs)}")
    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
