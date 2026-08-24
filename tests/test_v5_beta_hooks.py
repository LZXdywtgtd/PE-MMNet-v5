"""
v5-β 钩子集成测试（V5-β 适配回归测试）

覆盖：
- TrainerV5 接收 v5-β 参数（use_position_image/use_uv_inverse/data_source）
- OrderedKeypointLoss 接收 lambda_position_6d/lambda_uv_projection
- collate_v5 支持 position_6d 字段（条件输出）
- PatchDatasetV5 npz 缓存（V5B6）
- PatchDatasetV5 use_position_image 注入
- PositionExtractor L0-L4 五级降级（V5B4）
- crack_annotator save/load + auto_extract_crack（V5B3）
- team_train_v5.py --list-tasks 加载（V5B7）

详见：
- tasks/team_v5_alpha.json 中 V5B3/V5B4/V5B6/V5B7
- docs/v5_范围说明.md §11 v5-β 8 任务
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from training.trainer_v5 import TrainerV5, CSVHistory, set_seed
from training.ordered_kp_loss import (
    OrderedKeypointLoss, position_6d_loss, uv_projection_loss,
)
from data.patch_dataset_v5 import PatchDatasetV5, collate_v5
from data.position_extractor import (
    PositionExtractor, extract_L0, extract_L1, extract_L2,
    extract_L3, extract_L4, PositionInfo,
)
from tools.crack_annotator import (
    save_annotation, load_annotation, auto_extract_crack,
)
import team_train_v5


# 快速热曲线
FAST_PROFILE = {
    "ramp_up_c_per_min": 1260.0,
    "soak_temp_c": 1280.0,
    "soak_duration_min": 1.0,
    "cool_down_c_per_min": 1260.0,
}


def _safe_mkdtemp(prefix: str) -> str:
    """沙盒安全的临时目录"""
    base = PROJECT_ROOT / "logs" / "_test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    import uuid
    path = base / f"{prefix}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# ============================================================
#  V5-β 钩子参数
# ============================================================

def test_trainer_v5_accepts_beta_params():
    """TrainerV5 接受 v5-β 参数（默认值不改变行为）"""
    tmp = _safe_mkdtemp("trainer_beta_params")
    try:
        trainer = TrainerV5(
            variant="resnet18",
            n_samples=2,
            patch_size=64,
            batch_size=2,
            epochs=1,
            crack_stress_threshold_MPa=5.0,
            log_dir=tmp,
            thermal_profile=FAST_PROFILE,
            # V5-β 钩子（默认禁用）
            data_source="simulation",
            use_position_image=False,
            use_uv_inverse=False,
            lambda_position_6d=0.0,
            lambda_uv_projection=0.0,
        )
        # 默认值检查
        assert trainer.data_source == "simulation"
        assert trainer.use_position_image is False
        assert trainer.use_uv_inverse is False
        assert trainer.lambda_position_6d == 0.0
        assert trainer.lambda_uv_projection == 0.0
        assert trainer.cache_dir is None
        print(f"  [OK] trainer_v5_beta_params: 默认值正确")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_trainer_v5_warns_when_beta_misused():
    """data_source='simulation' + use_position_image=True 应触发 warning（不崩）"""
    tmp = _safe_mkdtemp("trainer_warn")
    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trainer = TrainerV5(
                variant="resnet18",
                n_samples=2,
                patch_size=64,
                batch_size=2,
                epochs=1,
                crack_stress_threshold_MPa=5.0,
                log_dir=tmp,
                thermal_profile=FAST_PROFILE,
                use_position_image=True,  # 应 warning
                data_source="simulation",
            )
            assert any("use_position_image" in str(wi.message) for wi in w), \
                f"应触发 warning，实际 {[str(wi.message) for wi in w]}"
            print(f"  [OK] trainer_v5_beta_warn: warning 已触发")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_loss_fn_accepts_position_6d():
    """OrderedKeypointLoss 接收 position_6d（None / 真值 两路径）"""
    loss_fn = OrderedKeypointLoss(
        lambda_position_6d=0.1,
        lambda_uv_projection=0.05,
    )
    B, K, N = 4, 12, 50
    pred_bbox = torch.rand(B, 4)
    pred_kpts = torch.rand(B, K, 2)
    true_bbox = torch.rand(B, 4)
    true_px = torch.rand(B, N, 2)

    # 路径1：None → 占位返回 0
    losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_px)
    assert losses["position_6d"].item() == 0.0, \
        f"None 时应为 0，实际 {losses['position_6d'].item()}"
    assert losses["uv_projection"].item() == 0.0
    # 总损失不应被 v5-β 项影响（lambda=0 也成立）
    print(f"  [OK] loss_fn_none_path: position_6d=0, uv_projection=0")

    # 路径2：真值 → Smooth L1 / L2
    true_pos6d = torch.rand(B, 6)
    pred_pos6d = torch.rand(B, 6)
    losses = loss_fn(
        pred_bbox, pred_kpts, true_bbox, true_px,
        true_position_6d=true_pos6d, pred_position_6d=pred_pos6d,
    )
    assert losses["position_6d"].item() >= 0
    print(f"  [OK] loss_fn_real_path: pos6d={losses['position_6d'].item():.4f}")


def test_collate_v5_position_6d_optional():
    """collate_v5 仅在所有样本含 position_6d 时输出该字段"""
    # 路径1：无 position_6d → 不输出
    batch = [
        {"x_1d": torch.zeros(10), "x_2d": torch.zeros(3, 8, 8),
         "true_bbox": torch.zeros(4),
         "true_keypoints": torch.zeros(5, 2),
         "true_crack_pixels": torch.zeros(3, 2)},
        {"x_1d": torch.zeros(10), "x_2d": torch.zeros(3, 8, 8),
         "true_bbox": torch.zeros(4),
         "true_keypoints": torch.zeros(5, 2),
         "true_crack_pixels": torch.zeros(3, 2)},
    ]
    out = collate_v5(batch)
    assert "true_position_6d" not in out, "默认不应输出"
    print(f"  [OK] collate_v5_no_position_6d: 字段缺失")

    # 路径2：全部含 position_6d → 输出 (B, 6)
    for b in batch:
        b["true_position_6d"] = torch.zeros(6)
    out = collate_v5(batch)
    assert "true_position_6d" in out
    assert out["true_position_6d"].shape == (2, 6)
    print(f"  [OK] collate_v5_with_position_6d: shape={tuple(out['true_position_6d'].shape)}")

    # 路径3：部分样本含 → 不输出（保守策略）
    batch[0].pop("true_position_6d")
    out = collate_v5(batch)
    assert "true_position_6d" not in out, "部分样本缺时不应输出"
    print(f"  [OK] collate_v5_partial: 字段缺失（保守）")


# ============================================================
#  V5B6 npz 缓存
# ============================================================

def test_patch_dataset_v5_npz_cache():
    """npz 缓存写入 + 读取一致性"""
    cache_dir = str(PROJECT_ROOT / "logs" / "_test_npz")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    try:
        # 第 1 次：写入
        ds1 = PatchDatasetV5(
            patch_size=64, n_samples=2, seed=42,
            thermal_profile=FAST_PROFILE,
            cache_dir=cache_dir,
            verbose=False,
        )
        files = os.listdir(cache_dir)
        assert len(files) == 1, f"应 1 个 .npz，实际 {len(files)}"
        cache_file = os.path.join(cache_dir, files[0])
        assert cache_file.endswith(".npz")
        print(f"  [OK] npz_cache_written: {os.path.basename(cache_file)}")

        # 第 2 次：读取 + 数据一致性
        ds2 = PatchDatasetV5(
            patch_size=64, n_samples=2, seed=42,
            thermal_profile=FAST_PROFILE,
            cache_dir=cache_dir,
            verbose=False,
        )
        s1 = ds1[0]
        s2 = ds2[0]
        assert torch.allclose(s1["x_1d"], s2["x_1d"])
        assert torch.allclose(s1["x_2d"], s2["x_2d"])
        print(f"  [OK] npz_cache_consistent: x_1d + x_2d 一致")

        # 第 3 次：use_position_image=True → 字段注入
        ds3 = PatchDatasetV5(
            patch_size=64, n_samples=2, seed=42,
            thermal_profile=FAST_PROFILE,
            use_position_image=True,
            verbose=False,
        )
        s3 = ds3[0]
        assert "true_position_6d" in s3
        assert s3["true_position_6d"].shape == (6,)
        print(f"  [OK] npz_with_position_6d: 字段注入成功")
    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


# ============================================================
#  V5B4 位置图像 5 级降级
# ============================================================

def test_position_extractor_levels():
    """L0-L4 各级返回 PositionInfo 且 validity_mask 符合预期"""
    # L4
    p = extract_L4()
    assert p.level == "L4" and not p.validity_mask.any()

    # L3
    p = extract_L3(cart_id=2, layer_idx=1)
    assert p.level == "L3" and p.validity_mask.all()
    assert p.position_6d.shape == (6,)

    # L2
    p = extract_L2(ceramic_count=3)
    assert p.level == "L2" and p.validity_mask[5].item() is True
    assert p.validity_mask[:5].sum().item() == 0

    # L1 stub（无 cv2 → 自动降级到 L3）
    fake_img = np.zeros((64, 64, 3), dtype=np.uint8)
    p = extract_L1(single_image=fake_img, kiln_visible=True)
    # 自动降级：可能是 L3 或 L1（取决于 cv2 是否在）
    assert p.level in ("L1", "L3"), f"应 L1 或 L3，实际 {p.level}"

    # L0 stub
    p = extract_L0(multi_view_images=[fake_img, fake_img])
    assert p.level in ("L0", "L3"), f"应 L0 或 L3，实际 {p.level}"

    print(f"  [OK] position_extractor_levels: 5 级返回正确")


def test_position_extractor_auto_mode():
    """auto 模式根据输入自动选最高级"""
    # 无图 → 选 L2（ceramic_count > 0）
    ext = PositionExtractor(mode="auto", ceramic_count=5)
    p = ext.extract()
    assert p.level in ("L2", "L3")
    # 有 1 张图 → 选 L1 (cv2 缺失 → 降级)
    ext = PositionExtractor(mode="auto", images=np.zeros((64, 64, 3), dtype=np.uint8))
    p = ext.extract()
    assert p.level in ("L1", "L3")
    print(f"  [OK] position_extractor_auto: 智能降级")


def test_position_extractor_to_device():
    """PositionInfo.to(device) 正确迁移"""
    p = extract_L3(cart_id=1, layer_idx=0)
    p_gpu = p.to(torch.device("cpu"))
    assert isinstance(p_gpu, PositionInfo)
    assert p_gpu.position_6d.device.type == "cpu"
    print(f"  [OK] position_info_to_device: 迁移成功")


# ============================================================
#  V5B3 裂纹标注
# ============================================================

def test_crack_annotator_save_load():
    """save/load 序列化一致性"""
    out_path = "logs/_test_ann_v5b3.json"
    if os.path.exists(out_path):
        os.remove(out_path)
    try:
        kpts = np.array([[10, 20], [30, 40], [50, 60], [70, 80]] * 3,
                        dtype=np.float32).reshape(12, 2)
        bbox = (15.0, 5.0, 80.0, 75.0)
        path = save_annotation(
            image_path="dummy.png", keypoints=kpts, bbox=bbox,
            ceramic_id="test_c01", output_path=out_path,
        )
        assert os.path.exists(path)

        ann = load_annotation(path)
        assert ann["ceramic_id"] == "test_c01"
        assert len(ann["crack_keypoints"]) == 12
        assert np.allclose(ann["crack_keypoints"], kpts)
        print(f"  [OK] crack_annotator_save_load: 12 点一致")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_crack_annotator_auto_extract():
    """auto_extract_crack 合成测试"""
    try:
        from skimage import feature
        H, W = 128, 128
        img = np.ones((H, W, 3), dtype=np.uint8) * 255
        # 画 S 形裂纹
        for i in range(20):
            y = int(20 + i * 4)
            x = int(64 + 20 * np.sin(i * 0.5))
            if 0 <= y < H and 0 <= x < W:
                img[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 0

        result = auto_extract_crack(img, min_keypoints=4, max_keypoints=8)
        assert result["keypoints"].shape[1] == 2
        assert len(result["bbox"]) == 4
        print(f"  [OK] crack_annotator_auto: K={len(result['keypoints'])}, "
              f"bbox={result['bbox']}")
    except ImportError:
        print(f"  [跳过] skimage 未装")


# ============================================================
#  V5B7 team_train_v5
# ============================================================

def test_team_train_v5_list_tasks():
    """team_train_v5 --list-tasks 加载 V5A*/V5B* 任务"""
    # 临时捕获 stdout
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            sys.argv = ["team_train_v5.py", "--list-tasks"]
            team_train_v5.main()
        except SystemExit:
            pass
    output = f.getvalue()
    # 应只列出 V5 前缀任务
    assert "V5A1" in output, f"应列出 V5A1"
    assert "V5B3" in output, f"应列出 V5B3"
    assert "V5B4" in output, f"应列出 V5B4"
    # 不应包含 v4 残留（A1/A2/.../C3）
    assert "\nA1    " not in output, f"不应含 v4 A1"
    assert "\nA2    " not in output, f"不应含 v4 A2"
    print(f"  [OK] team_train_v5_list_tasks: V5 任务过滤正确")


def test_team_train_v5_module_test():
    """team_train_v5.py 模块测试路由"""
    # 检查 _V5_MODULE_TEST_MAP 含预期模块
    expected_modules = {
        "simulator", "heads", "loss", "backbones", "trainer",
        "annotator", "position", "cache",
    }
    actual_modules = set(team_train_v5._V5_MODULE_TEST_MAP.keys())
    assert expected_modules.issubset(actual_modules), \
        f"缺失模块: {expected_modules - actual_modules}"
    print(f"  [OK] team_train_v5_module_test: {len(actual_modules)} 个模块路由")


# ============================================================
#  运行所有测试
# ============================================================

def run_all():
    tests = [
        # V5-β 钩子参数
        test_trainer_v5_accepts_beta_params,
        test_trainer_v5_warns_when_beta_misused,
        test_loss_fn_accepts_position_6d,
        test_collate_v5_position_6d_optional,
        # V5B6
        test_patch_dataset_v5_npz_cache,
        # V5B4
        test_position_extractor_levels,
        test_position_extractor_auto_mode,
        test_position_extractor_to_device,
        # V5B3
        test_crack_annotator_save_load,
        test_crack_annotator_auto_extract,
        # V5B7
        test_team_train_v5_list_tasks,
        test_team_train_v5_module_test,
    ]

    print("=" * 60)
    print(f"v5-β 钩子 + V5B3/V5B4/V5B6/V5B7 回归测试（共 {len(tests)} 项）")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())