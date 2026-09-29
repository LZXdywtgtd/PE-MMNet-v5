"""
v5-α 统一训练脚本（V5A5）

用法：
    python run_train_v5.py --variant resnet18 --epochs 10
    python run_train_v5.py --variant swin_yolo --epochs 100 --n_samples 1000
    python run_train_v5.py --variant detr --resume checkpoints/v5/detr/latest.pt

设计：
- 沿用 v4 的 CLI 风格（argparse + Tee 日志）
- 默认使用 v5-α 默认值（patch_size=256, min_kpts=8, max_kpts=16）
- 可选 GPU/CPU；自动选 batch size

详见：
- tasks/team_v5_alpha.json 中 V5A5
"""

from __future__ import annotations

import os
import sys
import argparse
import time

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 禁用缓冲
import functools
print = functools.partial(print, flush=True)

# UTF-8 兼容
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace',
            line_buffering=True,
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding='utf-8', errors='replace',
            line_buffering=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v5-α 训练入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 模型
    parser.add_argument(
        "--variant", type=str, default="resnet18",
        choices=["resnet18", "swin_yolo", "vit_yolo", "detr", "swin_yolo_patchtst"],
        help="模型变体",
    )
    parser.add_argument("--pretrained_2d", action="store_true",
                        help="使用 ImageNet 预训练权重")
    parser.add_argument("--no_gp", action="store_true", help="禁用 GP 模块")
    # 数据
    parser.add_argument("--n_samples", type=int, default=100,
                        help="数据集样本数（仿真生成）")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="patch 像素尺寸")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--min_kpts", type=int, default=8)
    parser.add_argument("--max_kpts", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=300)
    parser.add_argument("--crack_stress_threshold_MPa", type=float, default=50.0)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="仿真 npz 缓存目录（命中则零仿真）")
    parser.add_argument("--no_aug", action="store_true", help="禁用数据增强")
    # 训练
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--coverage_mode", type=str, default="chamfer",
                        choices=["chamfer", "hausdorff"])
    parser.add_argument("--M", type=int, default=200,
                        help="B 样条采样点数")
    parser.add_argument("--lambda_coverage", type=float, default=1.0)
    parser.add_argument("--lambda_bbox", type=float, default=1.0)
    parser.add_argument("--lambda_gp", type=float, default=0.1)
    parser.add_argument("--lambda_poisson", type=float, default=0.05)
    parser.add_argument("--save_every", type=int, default=1)
    # 设备 / 路径
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu（None = 自动）")
    parser.add_argument("--log_dir", type=str, default="logs/training_history/run_train_v5")
    parser.add_argument("--seed", type=int, default=42)
    # 续训
    parser.add_argument("--resume", type=str, default=None,
                        help="检查点路径（None = 全新训练）")
    # 热曲线（可选加速）
    parser.add_argument("--fast_thermal", action="store_true",
                        help="用 1 min 短周期（仅测试用）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 热曲线（可选加速）
    thermal_profile = None
    if args.fast_thermal:
        thermal_profile = {
            "ramp_up_c_per_min": 1260.0,
            "soak_temp_c": 1280.0,
            "soak_duration_min": 1.0,
            "cool_down_c_per_min": 1260.0,
        }

    # 延迟导入（启动加速）
    from training.trainer_v5 import TrainerV5

    trainer = TrainerV5(
        variant=args.variant,
        n_samples=args.n_samples,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        min_kpts=args.min_kpts,
        max_kpts=args.max_kpts,
        coverage_mode=args.coverage_mode,
        M=args.M,
        lambda_coverage=args.lambda_coverage,
        lambda_bbox=args.lambda_bbox,
        lambda_gp=args.lambda_gp,
        lambda_poisson=args.lambda_poisson,
        use_aug=not args.no_aug,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        thermal_profile=thermal_profile,
        crack_stress_threshold_MPa=args.crack_stress_threshold_MPa,
        save_every=args.save_every,
        cache_dir=args.cache_dir,
        model_kwargs={"pretrained_2d": args.pretrained_2d},
    )

    # 如果禁用 GP，覆盖 model
    if args.no_gp:
        from models import create_v5_model
        trainer.model = create_v5_model(
            args.variant,
            image_channels=3,
            pretrained_2d=args.pretrained_2d,
            use_gp=False,
            max_kpts=args.max_kpts,
            min_kpts=args.min_kpts,
        ).to(trainer.device)
        print("[INFO] GP 模块已禁用")

    print(f"\n{'='*60}")
    print(f"v5-α 训练启动")
    print(f"{'='*60}")
    print(f"  变体:        {args.variant}")
    print(f"  数据样本:    {args.n_samples}")
    print(f"  patch:       {args.patch_size}x{args.patch_size}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  epochs:      {args.epochs}")
    print(f"  lr:          {args.lr}")
    print(f"  关键点:      [{args.min_kpts}, {args.max_kpts}]")
    print(f"  coverage:    {args.coverage_mode}")
    print(f"  λ 权重:      cov={args.lambda_coverage} "
          f"bbox={args.lambda_bbox} "
          f"gp={args.lambda_gp} "
          f"poisson={args.lambda_poisson}")
    print(f"  设备:        {trainer.device}")
    print(f"  log_dir:     {args.log_dir}")
    print(f"  cache_dir:   {args.cache_dir or '无（现场仿真）'}")
    print(f"  续训:        {args.resume or '无（全新）'}")

    t0 = time.time()
    final = trainer.fit(resume_from=args.resume, verbose=True)
    dt = time.time() - t0

    print(f"\n{'='*60}")
    print(f"训练完成：耗时 {dt:.1f}s")
    print(f"  最终 train_loss: {final['last_train_loss']:.4f}")
    print(f"  最终 val_loss:   {final['last_val_loss']:.4f}")
    print(f"  最佳 val_loss:   {final['best_val_loss']:.4f}")
    print(f"  CSV 历史:        {trainer.csv_path}")
    print(f"  检查点:          {trainer.log_dir}/checkpoints/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()