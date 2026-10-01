# -*- coding: utf-8 -*-
"""v5-β 第一诊断任务：尾部 7% 可视化（13 个未通过样本，噪声 vs 真难裁决）。

augfix2 best.pt 上判据 4 逐样本未通过的样本（chamfer > REQ=0.0441），
逐样本画 pred 样条 vs GT 对比图，看是否有共性形态：
  有共性 → 主线 1（93%→100% 收尾）立项干预
  无共性 → 93% 是实际极限，主线 1 关闭

输出：output/v5b_diag_viz/tail7pct/（独立子文件夹，用户要求分文件夹）
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import functools
print = functools.partial(print, flush=True)
import numpy as np
import torch

from _diag_variance79 import VAL_CACHE, PX
from _diag_v5b_direction2 import cham_of, UPPER
from _viz_common import setup_font, gt_px

CKPT = "logs/training_history/v5a6_dprime_augfix2/checkpoints/best.pt"
OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz", "tail7pct")
REQ = UPPER * 1.5

os.makedirs(OUT, exist_ok=True)
plt = setup_font()
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import catmull_rom_spline_torch

m = create_v5_model("resnet18", image_channels=3, image_size=128,
                    pretrained_2d=False, use_gp=True, min_kpts=8,
                    max_kpts=16, spatial_head=False)
st = load_checkpoint(CKPT, m, map_location="cuda")
print(f"best.pt epoch={st.get('epoch')}")
m.eval().cuda()

d = np.load(VAL_CACHE)
n = d["x_2d"].shape[0]

# 1) 全量推理 + 判据 4 逐样本 chamfer
preds = []
with torch.no_grad():
    for i in range(0, n, 50):
        x2 = torch.from_numpy(d["x_2d"][i:i+50]).float().cuda()
        x1 = torch.from_numpy(d["x_1d"][i:i+50]).float().cuda()
        preds.append(m(x1, x2)["keypoints"].cpu().numpy())
preds = np.concatenate(preds)

cham, rows = {}, []
for i in range(n):
    if d["pixel_mask"][i].sum() < 2:
        continue
    px = gt_px(d, i)
    ch = cham_of(preds[i].astype(np.float64), px)
    cham[i] = ch
    rows.append((i, ch, px))

pass_n = sum(1 for _, c, _ in rows if c <= REQ)
print(f"正样本 {len(rows)}：通过 {pass_n}（{pass_n/len(rows)*100:.0f}%），"
      f"未通过 {len(rows)-pass_n}")
tail = sorted(rows, key=lambda r: -r[1])
tail = [r for r in tail if r[1] > REQ]
print(f"尾部样本 idx/chamfer(px)："
      f"{[(i, round(c*PX,1)) for i, c, _ in tail]}")

# 2) 逐样本四联图（x_2d 三通道并排 + GT mask 叠加 + pred/GT 对比）
def draw_sample(i, ch, px, path):
    x2 = d["x_2d"][i]  # (3,H,W) 归一化
    kpts_gt = d["true_keypoints"][i]
    nk = int(d["keypoint_mask"][i].sum())
    pk = torch.from_numpy(preds[i]).float().cuda().unsqueeze(0)
    sp = catmull_rom_spline_torch(pk, 200)[0].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for c, ax in zip(range(3), axes[:3]):
        ax.imshow(x2[c], cmap="inferno", vmin=0, vmax=1)
        ax.set_title(["ch0 温度场", "ch1 应力场", "ch2 热力图通道"][c], fontsize=10)
        ax.plot(px[:, 1]*PX, px[:, 0]*PX, ".", color="cyan", ms=0.6, alpha=0.7)
    ax = axes[3]
    ax.imshow(x2[1], cmap="gray", vmin=0, vmax=1)
    ax.plot(px[:, 1]*PX, px[:, 0]*PX, ".", color="lime", ms=1.2, label="GT 裂纹")
    valid_gt = kpts_gt[:nk]
    ax.plot(valid_gt[:, 1]*PX, valid_gt[:, 0]*PX, "o", mfc="none",
            mec="lime", ms=7, mew=1.2, label="GT kpts")
    pts = sp[:, ::-1] * PX
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, colors="red", linewidths=1.6,
                                     label="pred 样条"))
    ax.plot(preds[i][:, 1]*PX, preds[i][:, 0]*PX, "r.", ms=4, label="pred kpts")
    gyc, gxc = px[:, 0].mean()*PX, px[:, 1].mean()*PX
    pyc, pxc = preds[i][:, 0].mean()*PX, preds[i][:, 1].mean()*PX
    ax.plot([gxc, pxc], [gyc, pyc], "w--", lw=1, alpha=0.8)
    dxf, dyf = pxc-gxc, pyc-gyc
    ang = np.degrees(np.arctan2(dyf, dxf))
    cr = px[:, 1].max() - px[:, 1].min()
    cyy = px[:, 0].max() - px[:, 0].min()
    elong = max(cr, cyy) / max(min(cr, cyy), 1)
    nb = int(d["pixel_mask"][i].sum())
    ax.set_title(f"pred vs GT  chamfer={ch:.4f} ({ch*PX:.1f}px)"
                 f"{' ✗超阈' if ch > REQ else ''}\n"
                 f"质心偏移({dxf:+.0f},{dyf:+.0f})px 角度{ang:.0f}° "
                 f"| GT bbox {cr:.0f}x{cyy:.0f}px elong={elong:.1f} "
                 f"n_px={nb}", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    for ax in axes:
        ax.set_xlim(0, PX); ax.set_ylim(PX, 0); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"尾部样本 idx={i}  chamfer {ch:.4f} / 阈值 {REQ:.4f} "
                 f"(x{ch/REQ:.1f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=110)
    plt.close(fig)

summary = []
for i, ch, px in tail:
    p = os.path.join(OUT, f"tail_idx{i:04d}_ch{ch:.4f}.png")
    draw_sample(i, ch, px, p)
    x2 = d["x_2d"][i]
    gyc, gxc = px[:, 0].mean(), px[:, 1].mean()
    pyc, pxc = preds[i][:, 0].mean(), preds[i][:, 1].mean()
    cr = (px[:, 1].max()-px[:, 1].min())
    cyy = (px[:, 0].max()-px[:, 0].min())
    summary.append(dict(idx=int(i), cham=ch, cham_px=ch*PX,
                        dcx=(pxc-gxc)*PX, dcy=(pyc-gyc)*PX,
                        gt_w=cr*PX, gt_h=cyy*PX,
                        n_px=int(d["pixel_mask"][i].sum()),
                        contrast=float(x2[1].max()-x2[1].min()),
                        margin=float(min(gxc, gyc, 1-gxc, 1-gyc)*PX)))

print("\n===== 尾部共性初判（数字表）=====")
print(f"{'idx':>5} {'ch_px':>6} {'dcx':>6} {'dcy':>6} {'gt_w':>5} {'gt_h':>5} "
      f"{'n_px':>5} {'contrast':>8} {'margin_px':>9}")
for r in summary:
    print(f"{r['idx']:>5} {r['cham_px']:>6.1f} {r['dcx']:>+6.0f} {r['dcy']:>+6.0f} "
          f"{r['gt_w']:>5.0f} {r['gt_h']:>5.0f} {r['n_px']:>5} "
          f"{r['contrast']:>8.3f} {r['margin']:>9.1f}")

ws = np.array([r["gt_w"] for r in summary])
hs = np.array([r["gt_h"] for r in summary])
npx = np.array([r["n_px"] for r in summary])
ct = np.array([r["contrast"] for r in summary])
mg = np.array([r["margin"] for r in summary])
dc = np.hypot(np.array([r["dcx"] for r in summary]),
              np.array([r["dcy"] for r in summary]))
print("\n判读锚点：")
print(f"  GT 宽 p50={np.median(ws):.0f}px 高 p50={np.median(hs):.0f}px "
      f"n_px p50={np.median(npx):.0f}（全 val 中位参考 ~细裂纹）")
print(f"  应力对比度 p50={np.median(ct):.3f}（对照全 val p50）")
print(f"  贴边 margin p50={np.median(mg):.1f}px（<12px 即贴边）")
print(f"  质心偏移 p50={np.median(dc):.0f}px（位置错 vs 局部形状错）")
all_ct = np.array([float(d['x_2d'][i][1].max()-d['x_2d'][i][1].min())
                   for i, c, _ in rows])
all_mg = np.array([float(min(d['true_keypoints'][i][d['keypoint_mask'][i]][:, 0].mean(),
                             1-d['true_keypoints'][i][d['keypoint_mask'][i]][:, 0].mean()))*PX
                   for i, c, _ in rows])
print(f"  全 val 对照：contrast p50={np.median(all_ct):.3f}")
d.close()
print(f"\n图已存 {OUT}（{len(tail)} 张）")
