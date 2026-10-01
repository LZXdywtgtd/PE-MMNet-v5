# -*- coding: utf-8 -*-
"""增广器标记点自洽性测试：图像变换与标签变换是否同步（实测，不靠方向记忆）。

方法：x_2d 放单个亮点（argmax 可定位），kpts=[同一点]，
分别强制 hflip / vflip / rot90（概率写死 0/1），比较
  图像亮点的实际新位置 (argmax)
vs
  标签 kpts 的新位置。
一致 = 增广自洽；不一致 = 图像/标签错位（训练监督被污染）。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch

from training.data_aug_v5 import PatchAugmentorV5

H = W = 128


def one_test(name, **probs):
    aug = PatchAugmentorV5(noise_prob=0.0, **probs)
    x2 = torch.zeros(3, H, W)
    y0, x0 = 32, 80          # 非对称点，避免翻转后重合
    x2[0, y0, x0] = 1.0
    kpts = torch.tensor([[y0 / H, x0 / W]], dtype=torch.float32)
    pixels = kpts.clone()
    bbox = torch.tensor([x0 / W - 0.05, y0 / H - 0.05,
                         x0 / W + 0.05, y0 / H + 0.05], dtype=torch.float32)
    x1d = torch.zeros(16)

    torch.manual_seed(0)
    np.random.seed(0)
    import random as _r
    _r.seed(0)
    _, x2a, bba, ka, pa = aug(x1d, x2.clone(), bbox.clone(),
                              kpts.clone(), pixels.clone())

    img = np.argwhere(x2a[0].numpy() > 0.5)[0]     # (y', x') 像素
    lab = ka[0].numpy() * np.array([H, W])         # (y'', x'') 像素
    ok = bool(np.abs(img - lab).max() <= 2.0)
    bb = bba.numpy() * np.array([W, H, W, H])      # x1,y1,x2,y2 像素
    bb_ok = (bb[0] - 2 <= x0 / W * 0 + lab[1] <= bb[2] + 2
             and bb[1] - 2 <= lab[0] <= bb[3] + 2)
    print(f"[{name}] 图像亮点 -> (y,x)=({img[0]},{img[1]})  "
          f"标签 kpts -> (y,x)=({lab[0]:.0f},{lab[1]:.0f})  "
          f"kpts{'✓一致' if ok else '✗错位'}  "
          f"bbox含点{'✓' if bb_ok else '✗'}")
    return ok


def main():
    n_ok = 0
    n_ok += one_test("hflip", hflip_prob=1.0, vflip_prob=0.0, rot90_prob=0.0)
    n_ok += one_test("vflip", hflip_prob=0.0, vflip_prob=1.0, rot90_prob=0.0)
    n_ok += one_test("rot90", hflip_prob=0.0, vflip_prob=0.0, rot90_prob=1.0)
    print(f"\n{'ALL PASS' if n_ok == 3 else f'FAIL {3-n_ok}/3 错位'}"
          f"（{n_ok}/3 自洽）")
    return 0 if n_ok == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
