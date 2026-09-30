# -*- coding: utf-8 -*-
"""修后重训（v5a6_dprime_augfix）vs 旧 kd0.5：判据 3/4 + 按边混淆矩阵复测。

口径：val 197 正样本，判据 4 ratio=chamfer p50 / 0.0294（阈值 1.5），
逐样本通过率（v5-β 验收线 ≥80%），按边 failed 率 + 边混淆矩阵
（对称性收窄 = 立项文档新增辅助验收指标）。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch

from _diag_variance79 import (CKPT_BASE, VAL_CACHE, PX, POS_OK, FAIL_TH)
from _diag_v5b_direction2 import cham_of, UPPER
from models import create_v5_model
from training.trainer_v5 import load_checkpoint

CKPT_FIX = "logs/training_history/v5a6_dprime_augfix2/checkpoints/latest.pt"
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
NAMES = ("上", "下", "左", "右")


def edge_of(gy, gx):
    d = (gy, 127 - gy, gx, 127 - gx)
    return int(np.argmin(d))


def evaluate(ckpt, tag):
    m = create_v5_model("resnet18", image_channels=3, image_size=128,
                        pretrained_2d=False, use_gp=True, min_kpts=8,
                        max_kpts=16, spatial_head=False)
    load_checkpoint(ckpt, m, map_location="cuda")
    m.eval().cuda()
    d = np.load(VAL_CACHE)
    preds = []
    with torch.no_grad():
        for i in range(0, d["x_2d"].shape[0], 50):
            x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
            x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
            preds.append(m(x1, x2)["keypoints"].cpu().numpy())
    preds = np.concatenate(preds)

    chams, recs = [], []
    for i in range(d["x_2d"].shape[0]):
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
        if px.shape[0] < 2:
            continue
        ch = cham_of(preds[i].astype(np.float64), px)
        chams.append(ch)
        med = float(np.median(
            torch.cdist(torch.from_numpy(preds[i]).float().cuda(),
                        torch.from_numpy(px).float().cuda()
                        ).min(dim=-1).values.cpu().numpy()) * PX)
        gc = px.mean(axis=0) * PX
        pc = preds[i].mean(axis=0) * PX
        recs.append(dict(idx=i, med=med, cham=ch,
                         grp=("failed" if med > FAIL_TH else
                              "learned" if med < POS_OK else "mid"),
                         eg=edge_of(gc[0], gc[1]),
                         ep=edge_of(pc[0], pc[1])))
    chams = np.array(chams)
    n = len(chams)
    ratio = float(np.median(chams) / UPPER)
    pass_rate = float((chams <= UPPER * 1.5).mean())

    # 判据 3（inter_var 简化口径：pred 质心方差跨样本）
    cs = np.array([preds[r["idx"]].mean(axis=0) for r in recs])
    inter_var = float(cs.var(axis=0).sum())

    grp_cnt = {g: sum(r["grp"] == g for r in recs) for g in ("learned", "mid", "failed")}
    cm = np.zeros((4, 4), dtype=int)
    for r in recs:
        if r["grp"] == "failed":
            cm[r["eg"], r["ep"]] += 1
    f_opp = sum(cm[e, OPPOSITE[e]] for e in range(4))
    f_tot = cm.sum()
    print(f"\n[{tag}]")
    print(f"  判据4: chamfer p50={np.median(chams):.4f}  ratio={ratio:.2f} "
          f"({'PASS' if ratio <= 1.5 else 'FAIL'})  逐样本通过 {pass_rate*100:.0f}%")
    print(f"  判据3(近似): inter_var={inter_var:.4f}")
    print(f"  分组: learned {grp_cnt['learned']} ({grp_cnt['learned']/n*100:.0f}%)  "
          f"mid {grp_cnt['mid']}  failed {grp_cnt['failed']} ({grp_cnt['failed']/n*100:.0f}%)")
    print(f"  failed 边混淆: 对侧 {f_opp}/{f_tot} = {f_opp/max(f_tot,1)*100:.0f}%")
    for e in range(4):
        print(f"    GT{NAMES[e]}: " + " ".join(f"{cm[e,j]:>3d}" for j in range(4)))
    d.close()
    return dict(ratio=ratio, pass_rate=pass_rate, grp=grp_cnt, n=n,
                cm=cm.tolist(), opp=f_opp)


def main():
    old = evaluate(CKPT_BASE, "旧 kd0.5（增广 bug 时代）")
    new = evaluate(CKPT_FIX, "修后重训 augfix latest")
    print(f"\n对照总结:")
    print(f"  判据4 ratio: {old['ratio']:.2f} -> {new['ratio']:.2f}（阈值 1.5）")
    print(f"  逐样本通过: {old['pass_rate']*100:.0f}% -> {new['pass_rate']*100:.0f}%"
          f"（v5-β 验收线 80%）")
    print(f"  failed 率: {old['grp']['failed']/old['n']*100:.0f}% -> "
          f"{new['grp']['failed']/new['n']*100:.0f}%")
    print(f"  failed 对侧占比: {old['opp']} -> {new['opp']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
