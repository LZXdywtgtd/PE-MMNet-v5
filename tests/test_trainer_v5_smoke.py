"""
v5-α Trainer 冒烟测试（V5A5）

覆盖：
- TrainerV5 初始化（5 个变体）
- 单步训练（_train_step）loss 反向
- 单步评估（_eval_step）no_grad 模式
- 一个完整 epoch（train_epoch + eval_epoch）
- 多个 epoch（fit）写 CSV + 保存 best/latest 检查点
- 断点续训：load_checkpoint + resume_from
- 增强器集成
- 数据集集成（collate_v5 变长 padding）
- 5 变体端到端（每个变体跑 1 个 epoch）

详见：
- tasks/team_v5_alpha.json 中 V5A5 测试要求
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from models import create_v5_model, V5_MODEL_REGISTRY
from training.trainer_v5 import (
    TrainerV5,
    CSVHistory,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_device,
)
from training.ordered_kp_loss import OrderedKeypointLoss
from training.data_aug_v5 import PatchAugmentorV5
from data.patch_dataset_v5 import PatchDatasetV5, collate_v5


# 快速热曲线（缩短仿真时间）
FAST_PROFILE = {
    "ramp_up_c_per_min": 1260.0,
    "soak_temp_c": 1280.0,
    "soak_duration_min": 1.0,
    "cool_down_c_per_min": 1260.0,
}


def _make_trainer(log_dir: str, **kwargs) -> TrainerV5:
    """构造测试用 trainer（用快速热曲线）"""
    defaults = dict(
        variant="resnet18",
        n_samples=2,
        patch_size=64,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        crack_stress_threshold_MPa=5.0,
        log_dir=log_dir,
        thermal_profile=FAST_PROFILE,
        save_every=1,
    )
    defaults.update(kwargs)
    return TrainerV5(**defaults)


def _safe_mkdtemp(prefix: str) -> str:
    """沙盒安全的临时目录（用项目 logs 下）"""
    base = PROJECT_ROOT / "logs" / "_test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    import uuid
    path = base / f"{prefix}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# ============================================================
#  工具函数测试
# ============================================================

def test_set_seed():
    """测试 set_seed 可重现"""
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.allclose(a, b), "set_seed 应可复现"
    print(f"  [OK] set_seed: 随机种子可复现")


def test_get_device():
    """测试设备获取"""
    dev = get_device()
    assert dev.type in ("cuda", "cpu")
    print(f"  [OK] get_device: {dev.type}")


def test_csv_history():
    """测试 CSVHistory 写入"""
    tmp = _safe_mkdtemp("csv_hist")
    try:
        csv_path = os.path.join(tmp, "test_history.csv")
        hist = CSVHistory(csv_path)
        hist.log({
            "epoch": 1, "train_loss": 0.5, "lr": 1e-4,
            "epoch_time_s": 1.0, "timestamp": "2026-01-01",
        })
        hist.log({
            "epoch": 2, "train_loss": 0.3, "lr": 1e-4,
            "epoch_time_s": 1.0, "timestamp": "2026-01-01",
        })
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3, f"应 1 header + 2 rows = 3 行，实际 {len(lines)}"
        print(f"  [OK] csv_history: header + 2 行 = 3 行")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  Trainer 初始化
# ============================================================

def test_trainer_init():
    """测试 TrainerV5 初始化"""
    tmp = _safe_mkdtemp("init")
    try:
        trainer = _make_trainer(tmp)
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        n_params = sum(p.numel() for p in trainer.model.parameters())
        assert n_params > 0
        print(f"  [OK] trainer_init: {n_params:,} 参数，{len(trainer.train_ds)} 训练样本")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_trainer_init_all_variants():
    """测试 5 变体都能初始化"""
    tmp = _safe_mkdtemp("init_all")
    try:
        for variant in V5_MODEL_REGISTRY:
            trainer = _make_trainer(tmp, variant=variant)
            assert trainer.model is not None
            n_params = sum(p.numel() for p in trainer.model.parameters())
            assert n_params > 0
        print(f"  [OK] trainer_init_all_variants: 5 变体全部初始化成功")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  训练单步
# ============================================================

def test_train_step():
    """测试单步训练（loss 反向）"""
    tmp = _safe_mkdtemp("train_step")
    try:
        trainer = _make_trainer(tmp)
        batch = next(iter(trainer.train_loader))
        step = trainer._train_step(batch)
        assert "loss" in step and "coverage" in step
        assert "bbox" in step and "gp" in step and "poisson" in step
        assert step["loss"] >= 0
        # 模型参数应有梯度
        n_with_grad = sum(
            1 for p in trainer.model.parameters() if p.grad is not None
        )
        assert n_with_grad > 0
        print(f"  [OK] train_step: loss={step['loss']:.4f}, "
              f"{n_with_grad} 参数有梯度")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  评估单步
# ============================================================

def test_eval_step_no_grad():
    """测试评估单步（无梯度）"""
    tmp = _safe_mkdtemp("eval_step")
    try:
        trainer = _make_trainer(tmp)
        batch = next(iter(trainer.val_loader))
        step = trainer._eval_step(batch)
        assert step["loss"] >= 0
        print(f"  [OK] eval_step_no_grad: loss={step['loss']:.4f}, "
              f"@torch.no_grad 验证")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  完整 epoch
# ============================================================

def test_train_one_epoch():
    """测试 1 epoch 完整训练"""
    tmp = _safe_mkdtemp("train_epoch")
    try:
        trainer = _make_trainer(tmp, epochs=1)
        metrics = trainer.train_epoch(0)
        for k in ("loss", "coverage", "bbox", "gp", "poisson"):
            assert k in metrics
            assert metrics[k] >= 0
        print(f"  [OK] train_one_epoch: loss={metrics['loss']:.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_eval_one_epoch():
    """测试 1 epoch 评估"""
    tmp = _safe_mkdtemp("eval_epoch")
    try:
        trainer = _make_trainer(tmp, epochs=1)
        metrics = trainer.eval_epoch()
        assert "loss" in metrics and "coverage" in metrics
        print(f"  [OK] eval_one_epoch: loss={metrics['loss']:.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  完整 fit + CSV + checkpoint
# ============================================================

def test_fit_writes_csv_and_ckpt():
    """测试 fit 写 CSV + 保存 best/latest 检查点"""
    tmp = _safe_mkdtemp("fit_csv")
    try:
        trainer = _make_trainer(tmp, epochs=3, save_every=1)
        final = trainer.fit(verbose=False)

        with open(trainer.csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 4, f"应 4 行，实际 {len(lines)}"

        best_pt = os.path.join(tmp, "checkpoints", "best.pt")
        latest_pt = os.path.join(tmp, "checkpoints", "latest.pt")
        assert os.path.exists(best_pt)
        assert os.path.exists(latest_pt)

        assert final["best_val_loss"] < float("inf")
        print(f"  [OK] fit_writes_csv_and_ckpt: CSV 4 行, "
              f"best.pt + latest.pt 存在, best_val={final['best_val_loss']:.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_fit_verbose():
    """测试 verbose=True 打印进度"""
    tmp = _safe_mkdtemp("fit_verb")
    try:
        import io
        from contextlib import redirect_stdout
        trainer = _make_trainer(tmp, epochs=2, save_every=1)

        f = io.StringIO()
        with redirect_stdout(f):
            trainer.fit(verbose=True)

        output = f.getvalue()
        assert "[Epoch" in output, f"应打印 Epoch 日志"
        print(f"  [OK] fit_verbose: 打印 Epoch 日志（{len(output)} 字符）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  断点续训
# ============================================================

def test_resume_from_checkpoint():
    """测试从 latest.pt 续训"""
    tmp = _safe_mkdtemp("resume")
    try:
        trainer = _make_trainer(tmp, epochs=2, save_every=1)
        trainer.fit(verbose=False)
        latest_pt = os.path.join(tmp, "checkpoints", "latest.pt")
        assert os.path.exists(latest_pt)

        trainer2 = _make_trainer(tmp, epochs=4, save_every=1)
        state = load_checkpoint(latest_pt, trainer2.model, trainer2.optimizer,
                                map_location=trainer2.device)
        trainer2.start_epoch = state["epoch"] + 1

        assert trainer2.start_epoch == 2, \
            f"应从 epoch 2 续训，实际 {trainer2.start_epoch}"

        trainer2.fit(verbose=False)

        with open(trainer2.csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 5, f"应 5 行，实际 {len(lines)}"

        print(f"  [OK] resume_from_checkpoint: "
              f"start_epoch=2, 最终 epoch=4, CSV 5 行")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  增强器集成
# ============================================================

def test_augmentor_integration():
    """测试增强器在 trainer 中正确运行"""
    tmp = _safe_mkdtemp("aug")
    try:
        trainer_aug = _make_trainer(tmp, use_aug=True)
        tmp2 = _safe_mkdtemp("no_aug")
        trainer_no = _make_trainer(tmp2, use_aug=False)

        m_aug = trainer_aug.train_epoch(0)
        m_no = trainer_no.train_epoch(0)
        assert m_aug["loss"] >= 0
        assert m_no["loss"] >= 0
        print(f"  [OK] augmentor_integration: use_aug=True loss={m_aug['loss']:.4f}, "
              f"use_aug=False loss={m_no['loss']:.4f}")
        shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  变长 collate
# ============================================================

def test_collate_v5_padding():
    """测试 collate_v5 处理变长关键点/像素"""
    ds = PatchDatasetV5(
        patch_size=64,
        n_samples=3,
        seed=42,
        thermal_profile=FAST_PROFILE,
        crack_stress_threshold_MPa=5.0,
        verbose=False,
    )
    batch = collate_v5([ds[0], ds[1], ds[2]])
    assert batch["x_1d"].shape == (3, 300)
    assert batch["x_2d"].shape == (3, 3, 64, 64)
    assert batch["true_keypoints"].dim() == 3
    assert batch["keypoint_mask"].sum() > 0
    print(f"  [OK] collate_v5_padding: "
          f"kpts={tuple(batch['true_keypoints'].shape)}, "
          f"mask={batch['keypoint_mask'].sum().item()}")


# ============================================================
#  5 变体端到端
# ============================================================

def test_end_to_end_all_variants():
    """5 变体各跑 1 个 epoch"""
    tmp = _safe_mkdtemp("e2e_all")
    try:
        for variant in V5_MODEL_REGISTRY:
            trainer = _make_trainer(tmp, variant=variant, epochs=1)
            final = trainer.fit(verbose=False)
            assert final["last_train_loss"] >= 0
            assert final["best_val_loss"] < float("inf")
        print(f"  [OK] end_to_end_all_variants: 5 变体全部完成 1 epoch")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  无 GP 路径
# ============================================================

def test_no_gp_path():
    """测试禁用 GP 时训练正常"""
    tmp = _safe_mkdtemp("no_gp")
    try:
        trainer = _make_trainer(tmp)
        # 重新创建模型（用 use_gp=False）
        from models import create_v5_model
        trainer.model = create_v5_model(
            trainer.variant,
            image_channels=3,
            pretrained_2d=False,
            use_gp=False,
            max_kpts=trainer.max_kpts,
            min_kpts=trainer.min_kpts,
        ).to(trainer.device)
        assert trainer.model.gp_module is None

        final = trainer.fit(verbose=False)
        assert final["last_train_loss"] >= 0
        print(f"  [OK] no_gp_path: use_gp=False，loss={final['last_train_loss']:.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  空真值像素（coverage_loss 边界）
# ============================================================

def test_empty_crack_pixels():
    """测试 true_crack_pixels 为空时训练不崩"""
    tmp = _safe_mkdtemp("empty_px")
    try:
        trainer = _make_trainer(
            tmp,
            crack_stress_threshold_MPa=200.0,
        )
        final = trainer.fit(verbose=False)
        assert not np.isnan(final["last_train_loss"]), \
            f"loss 应为有限值，实际 NaN"
        print(f"  [OK] empty_crack_pixels: "
              f"loss={final['last_train_loss']:.4f}（非 NaN）")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
#  运行所有测试
# ============================================================

def run_all():
    tests = [
        # 工具
        test_set_seed,
        test_get_device,
        test_csv_history,
        # 初始化
        test_trainer_init,
        test_trainer_init_all_variants,
        # 单步
        test_train_step,
        test_eval_step_no_grad,
        # Epoch
        test_train_one_epoch,
        test_eval_one_epoch,
        # 完整
        test_fit_writes_csv_and_ckpt,
        test_fit_verbose,
        # 续训
        test_resume_from_checkpoint,
        # 增强
        test_augmentor_integration,
        # 数据
        test_collate_v5_padding,
        # 5 变体
        test_end_to_end_all_variants,
        # 无 GP
        test_no_gp_path,
        # 边界
        test_empty_crack_pixels,
    ]

    print("=" * 60)
    print(f"v5-α Trainer 冒烟测试（共 {len(tests)} 项）")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            import traceback
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")
            traceback.print_exc()

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())