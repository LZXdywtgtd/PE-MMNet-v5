"""
v5-α 物理因果链展示脚本（V5A6 子任务）

目的：跑 1 个变体（resnet18）5 epoch，输出：
1. 训练历史 CSV
2. 物理因果链 4 联图（温度场 → 应力场 → 裂纹 mask → 模型预测轨迹）

回应导师："AI 学习需要我们能推得出来" —— 模型学习的是物理链条，不是黑箱。

用法：
    python _v5a6_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 尝试找到系统中文字体
def _find_cjk_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        r"C:\Windows\Fonts\simfang.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return font_manager.FontProperties(fname=path)
    # 退回 DejaVu Sans（无中文支持）
    return None

_CJK_FONT = _find_cjk_font()
if _CJK_FONT is not None:
    plt.rcParams['font.family'] = _CJK_FONT.get_name()
    plt.rcParams['axes.unicode_minus'] = False

from data.patch_simulator_v5 import PatchSimulator
from data.patch_dataset_v5 import PatchDatasetV5, collate_v5
from models import create_v5_model
from training.trainer_v5 import TrainerV5


# 快速热曲线（1 min）
FAST_PROFILE = {
    "ramp_up_c_per_min": 1260.0,
    "soak_temp_c": 1280.0,
    "soak_duration_min": 1.0,
    "cool_down_c_per_min": 1260.0,
}


def run_physics_demo(
    variant: str = "resnet18",
    epochs: int = 5,
    n_samples: int = 4,
    patch_size: int = 64,
    output_dir: str = "logs/training_history/v5a6_demo",
    seed: int = 42,
):
    print(f"\n{'='*60}")
    print(f"v5-α 物理因果链展示（{variant} × {epochs} epoch）")
    print(f"{'='*60}\n")

    # ============ Step 1: 跑训练 ============
    print("[Step 1/3] 训练 5 epoch ...")
    trainer = TrainerV5(
        variant=variant,
        n_samples=n_samples,
        patch_size=patch_size,
        batch_size=2,
        epochs=epochs,
        lr=1e-3,
        min_kpts=8,
        max_kpts=16,
        coverage_mode="chamfer",
        M=200,
        lambda_coverage=1.0,
        lambda_bbox=1.0,
        lambda_gp=0.1,
        lambda_poisson=0.05,
        use_aug=True,
        seed=seed,
        crack_stress_threshold_MPa=5.0,
        log_dir=output_dir,
        thermal_profile=FAST_PROFILE,
        save_every=1,
        model_kwargs={"use_gp": True},
    )

    final = trainer.fit(verbose=True, resume_from=None)
    print(f"\n训练完成：train_loss={final['last_train_loss']:.4f}, "
          f"val_loss={final['last_val_loss']:.4f}\n")

    # ============ Step 2: 取 1 个样本做因果链图 ============
    print("[Step 2/3] 生成物理因果链展示图 ...")
    sim = PatchSimulator(
        patch_size=patch_size,
        thermal_profile=FAST_PROFILE,
        crack_stress_threshold_MPa=5.0,
        seed=seed + 100,  # 用另一个 seed 保证新样本
    )
    sample = sim.simulate(verbose=False)
    print(f"  样本元数据: peak_stress={sample['metadata']['peak_stress_MPa']:.1f} MPa, "
          f"K={sample['metadata']['n_keypoints']}")

    # ============ Step 3: 加载 best.pt，跑模型预测 ============
    print("[Step 3/3] 用训练好的模型预测 ...")
    best_ckpt_path = os.path.join(output_dir, "checkpoints", "best.pt")
    if not os.path.exists(best_ckpt_path):
        print(f"  WARN: best.pt 不存在，用 latest.pt")
        best_ckpt_path = os.path.join(output_dir, "checkpoints", "latest.pt")

    state = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    model = create_v5_model(
        variant,
        image_channels=3,
        pretrained_2d=False,
        use_gp=True,
        min_kpts=8,
        max_kpts=16,
    ).cuda()
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # 构造输入
    x_1d = torch.zeros(1, 300).cuda()  # 1D 时序（demo 用 0）
    x_2d_np = np.stack([
        sample["temperature_field"][-1],    # 最终温度场
        sample["stress_field"][-1] / 1e8,   # 归一化
        sample["heatmap"],
    ], axis=0).astype(np.float32)  # (3, H, W)
    x_2d = torch.from_numpy(x_2d_np).unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(x_1d, x_2d)

    pred_bbox = out["bbox"][0].cpu().numpy()  # (4,)
    pred_kpts = out["keypoints"][0].cpu().numpy()  # (K, 2)
    pred_validity = out["validity"][0].cpu().numpy().astype(bool)  # (K,)
    pred_K = int(out["K"][0].cpu().item())

    # 只取有效关键点
    valid_kpts = pred_kpts[pred_validity][:pred_K]
    print(f"  模型预测 K={pred_K}, bbox={pred_bbox}")

    # ============ 画 4 联图 ============
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"PE-MMNet v5-α 物理因果链（{variant} @ {epochs} epoch）\n"
        f"温度场 → 应力场 → 裂纹 GT → 模型预测",
        fontsize=14, fontweight="bold",
    )

    H, W = patch_size, patch_size

    # 1. 温度场
    im0 = axes[0, 0].imshow(sample["temperature_field"][-1], cmap="hot")
    axes[0, 0].set_title(f"1. 温度场（最终时刻）\nmax={sample['temperature_field'][-1].max():.0f}°C")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # 2. 应力场
    im1 = axes[0, 1].imshow(sample["stress_field"][-1] / 1e6, cmap="viridis")
    axes[0, 1].set_title(f"2. 应力场（最终时刻）\nmax={sample['stress_field'][-1].max()/1e6:.1f} MPa")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # 3. 裂纹 mask + GT 关键点
    axes[0, 2].imshow(sample["crack_mask"], cmap="gray")
    gt_kpts = sample["crack_keypoints"]
    axes[0, 2].scatter(gt_kpts[:, 1], gt_kpts[:, 0], c="red", s=40, label="GT")
    for i, (y, x) in enumerate(gt_kpts):
        axes[0, 2].annotate(str(i), (x, y), color="red", fontsize=8)
    gt_bbox = sample["crack_bbox"]
    rect = plt.Rectangle((gt_bbox[0]*W, gt_bbox[1]*H),
                          (gt_bbox[2]-gt_bbox[0])*W,
                          (gt_bbox[3]-gt_bbox[1])*H,
                          linewidth=2, edgecolor="lime", facecolor="none")
    axes[0, 2].add_patch(rect)
    axes[0, 2].set_title(f"3. GT：裂纹 mask + 关键点\nK={len(gt_kpts)} 个关键点")
    axes[0, 2].legend(loc="upper right")
    axes[0, 2].axis("off")

    # 4. 模型预测（叠加温度场作为底图）
    axes[1, 0].imshow(sample["temperature_field"][-1], cmap="hot", alpha=0.5)
    axes[1, 0].imshow(sample["crack_mask"], cmap="gray", alpha=0.3)
    axes[1, 0].scatter(valid_kpts[:, 1]*W, valid_kpts[:, 0]*H, c="cyan", s=40, label="Pred")
    for i, (y, x) in enumerate(valid_kpts):
        axes[1, 0].annotate(str(i), (x*W, y*H), color="cyan", fontsize=8)
    rect = plt.Rectangle((pred_bbox[0]*W, pred_bbox[1]*H),
                          (pred_bbox[2]-pred_bbox[0])*W,
                          (pred_bbox[3]-pred_bbox[1])*H,
                          linewidth=2, edgecolor="cyan", facecolor="none")
    axes[1, 0].add_patch(rect)
    axes[1, 0].set_title(f"4. 模型预测（{epochs} epoch 后）\nK={pred_K}, bbox={pred_bbox.round(2)}")
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].axis("off")

    # 5. 预测 vs GT 对比
    axes[1, 1].imshow(sample["crack_mask"], cmap="gray", alpha=0.5)
    axes[1, 1].scatter(gt_kpts[:, 1], gt_kpts[:, 0], c="red", s=40, label="GT", alpha=0.7)
    axes[1, 1].scatter(valid_kpts[:, 1]*W, valid_kpts[:, 0]*H, c="cyan", s=40, label="Pred", alpha=0.7)
    axes[1, 1].plot(valid_kpts[:, 1]*W, valid_kpts[:, 0]*H, "c-", alpha=0.5, lw=2)
    axes[1, 1].plot(gt_kpts[:, 1], gt_kpts[:, 0], "r-", alpha=0.5, lw=2)
    axes[1, 1].set_title("5. GT vs Pred 对比")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].axis("off")

    # 6. Loss 曲线
    import pandas as pd
    csv_path = os.path.join(output_dir, "training_history.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        axes[1, 2].plot(df["epoch"], df["train_loss"], "b-o", label="train_loss")
        axes[1, 2].plot(df["epoch"], df["val_loss"], "r-o", label="val_loss")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Loss")
        axes[1, 2].set_title("6. Loss 曲线")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "physics_causal_chain.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\n  保存到：{save_path}")

    # 画训练历史单独图
    if os.path.exists(csv_path):
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        df = pd.read_csv(csv_path)
        axes2[0].plot(df["epoch"], df["train_loss"], "b-o", label="train")
        axes2[0].plot(df["epoch"], df["val_loss"], "r-o", label="val")
        axes2[0].set_title(f"{variant} Total Loss")
        axes2[0].set_xlabel("Epoch")
        axes2[0].set_ylabel("Loss")
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        # 各分量 loss（如果有）
        loss_cols = [c for c in df.columns if c.endswith("_loss") and c not in ("train_loss", "val_loss")]
        if loss_cols:
            for c in loss_cols:
                axes2[1].plot(df["epoch"], df[c], "-o", label=c)
            axes2[1].set_title(f"{variant} Loss 分量")
            axes2[1].set_xlabel("Epoch")
            axes2[1].set_ylabel("Loss")
            axes2[1].legend()
            axes2[1].grid(True, alpha=0.3)
        else:
            axes2[1].axis("off")

        plt.tight_layout()
        loss_plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(loss_plot_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  保存到：{loss_plot_path}")

    print(f"\n{'='*60}")
    print(f"完成。打开 {save_path} 查看物理因果链。")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_physics_demo(
        variant="resnet18",
        epochs=5,
        n_samples=4,
        patch_size=64,
    )