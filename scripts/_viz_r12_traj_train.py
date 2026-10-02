# -*- coding: utf-8 -*-
"""Round12 工作项 3B（训练侧）：真梯度下降轨迹 20ep 重训。

最小 wrapper：构造 TrainerV5（口径完全对齐立项文档 §八 / 收官 run
v5b_s3_gtfix_dp），手动跑 20 个 epoch 的 train_epoch+eval_epoch 裸序列
（trainer 无 scheduler/AMP/梯度裁剪，fit 循环即此序列），每 epoch 存
traj_ckpts/ep{N:03d}.pt 供轨迹可视化。不修改任何现有文件。

可复现性验证：同 seed 下前 20 行 CSV 应复现收官 run 前 20 行
（对照并打印差异；失败即停——纪律第 6 条）。

用法：PYTHONIOENCODING=utf-8 python scripts/_viz_r12_traj_train.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import csv
import time

from training.trainer_v5 import TrainerV5, save_checkpoint

OUT = os.path.join(PROJECT_ROOT, "output", "v5b_diag_viz",
                   "round12_param_trajectory")
CKPT_DIR = os.path.join(OUT, "traj_ckpts")
os.makedirs(CKPT_DIR, exist_ok=True)

REF_RUN = "v5b_s3_gtfix_dp"
N_EPOCHS = 20

trainer = TrainerV5(
    variant="resnet18",
    n_samples=1000,
    patch_size=128,
    batch_size=4,
    epochs=N_EPOCHS,
    lr=1e-4,
    min_kpts=8,
    max_kpts=16,
    coverage_mode="chamfer",
    M=200,
    lambda_coverage=1.0, lambda_bbox=1.0, lambda_gp=0.1,
    lambda_poisson=0.05, lambda_kpt_direct=0.5,
    lambda_ordered=0.0, lambda_validity=0.0, lambda_heatmap=0.0,
    spatial_head=False,
    use_aug=True,
    seed=424242,
    train_seed=424242,
    log_dir=os.path.join(PROJECT_ROOT, "logs", "training_history",
                         "v5b_r12_traj20"),
    thermal_profile={"ramp_up_c_per_min": 1260.0, "soak_temp_c": 1280.0,
                     "soak_duration_min": 1.0,
                     "cool_down_c_per_min": 1260.0},
    crack_stress_threshold_MPa=60.0,
    save_every=N_EPOCHS,  # latest 只在末尾写一次，best 按口径照写
    cache_dir=os.path.join(PROJECT_ROOT, "logs", "sim_cache"),
    data_variant="dprime",
    model_kwargs={"pretrained_2d": False},
)
print(f"trainer 就绪 device={trainer.device}  "
      f"train_cache={os.path.basename(str(trainer.train_ds._cache_file))}")

# ---------- 20ep 训练（fit 的裸序列 + per-epoch checkpoint）----------
rows = []
for epoch in range(N_EPOCHS):
    t0 = time.time()
    tr = trainer.train_epoch(epoch)
    va = trainer.eval_epoch()
    dt = time.time() - t0
    row = {"epoch": epoch + 1,
           "train_loss": tr["loss"], "val_loss": va["loss"],
           "val_coverage": va["coverage_pos"], "time_s": dt}
    rows.append(row)
    print(f"[ep {epoch+1:3d}/{N_EPOCHS}] train={tr['loss']:.6f} "
          f"val={va['loss']:.6f} cov_pos={va['coverage_pos']:.6f} "
          f"({dt:.1f}s)")
    save_checkpoint(
        os.path.join(CKPT_DIR, f"ep{epoch+1:03d}.pt"),
        trainer.model, trainer.optimizer, epoch=epoch,
        best_val_loss=float("inf"),
        history_csv=trainer.csv_path,
        extras={"variant": "resnet18",
                "round12_traj": True, "epoch_1based": epoch + 1})

with open(os.path.join(OUT, "traj_history.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                      "val_coverage", "time_s"])
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                    for k, v in r.items()})

# ---------- 可复现性验证：对照收官 run 前 20 行 ----------
ref_path = os.path.join(PROJECT_ROOT, "logs", "training_history",
                        REF_RUN, "training_history.csv")
with open(ref_path, newline="", encoding="utf-8") as f:
    ref = list(csv.DictReader(f))[:N_EPOCHS]
print("\n---- 可复现性对照（wrapper vs 收官 run 前 20 行）----")
max_dt = 0.0
for r, rr in zip(rows, ref):
    dt_tl = abs(r["train_loss"] - float(rr["train_loss"]))
    dt_vl = abs(r["val_loss"] - float(rr["val_loss"]))
    max_dt = max(max_dt, dt_tl, dt_vl)
    flag = "OK" if max(dt_tl, dt_vl) < 1e-6 else "DIFF"
    print(f"ep{r['epoch']:3d}  dTrain={dt_tl:.2e}  dVal={dt_vl:.2e}  {flag}")
print(f"max |diff| = {max_dt:.3e}  "
      f"({'REPRODUCED' if max_dt < 1e-6 else 'MISMATCH——停，写 log 待拍板'})")
print("DONE round12 工作项3B 训练侧")
