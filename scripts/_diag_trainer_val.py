# -*- coding: utf-8 -*-
"""最终裁决：用训练同款 TrainerV5 构造 + eval_epoch()，对比离线复算。

若 trainer val=0.02 而离线=0.55 → trainer val 管线有问题（数据不同）。
若两者一致 0.55 → 训练时 CSV 数字假（保存/加载/续训问题）。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import torch

from models import create_v5_model
from training.trainer_v5 import TrainerV5, load_checkpoint

CKPT = "logs/training_history/v5a6_dprime_augfix/checkpoints/latest.pt"


def main():
    trainer = TrainerV5(
        variant="resnet18", n_samples=1000, patch_size=128, batch_size=32,
        epochs=150, seed=424242,
        crack_stress_threshold_MPa=60.0,
        thermal_profile={"ramp_up_c_per_min": 1260.0,
                         "soak_temp_c": 1280.0,
                         "soak_duration_min": 1.0,
                         "cool_down_c_per_min": 1260.0},
        log_dir="logs/training_history/v5a6_dprime_augfix",
        device="cuda",
    )
    load_checkpoint(CKPT, trainer.model, map_location="cuda")
    m = trainer.eval_epoch()
    print(f"\ntrainer 同款 eval_epoch: loss={m['loss']:.4f}  "
          f"coverage={m['coverage']:.4f}")
    print(f"训练 CSV ep150 val_coverage = 0.021843")
    print(f"离线复算（_diag_cov_discrepancy）= 0.5776")

    # 附加检查：val 集正样本数
    n_pos = sum(1 for i in range(len(trainer.val_ds))
                if trainer.val_ds._cache is not None and
                trainer.val_ds._cache[i]["true_crack_pixels"].size(0) > 0)
    print(f"trainer val_ds 正样本数 = {n_pos}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
