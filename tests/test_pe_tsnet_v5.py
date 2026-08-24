"""
v5-α 5 变体模型单元测试（V5A4）

覆盖：
- 5 个模型类：PETSNetMultimodalV5 / SwinYOLOFPNV5 / ViTYOLOFPNV5 / DETRStyleV5 / SwinYOLOFPNWithPatchTSTV5
- 工厂函数 create_v5_model
- 形状正确性
- 值域约束
- 反向传播（所有变体）
- GP 模块集成
- use_gp=False 路径
- 参数验证（无效 image_channels / min_kpts / max_kpts）

详见：
- tasks/team_v5_alpha.json 中 V5A4 测试要求
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models import (
    PETSNetMultimodalV5,
    SwinYOLOFPNV5,
    ViTYOLOFPNV5,
    DETRStyleV5,
    SwinYOLOFPNWithPatchTSTV5,
    create_v5_model,
    V5_MODEL_REGISTRY,
)
from training.ordered_kp_loss import OrderedKeypointLoss


# ============================================================
#  公共测试工具
# ============================================================

# 不下载预训练权重的通用 kwargs
COMMON_KWARGS = dict(
    seq_len=300,
    image_channels=3,
    pretrained_2d=False,  # 测试时不联网
    use_gp=True,
)

BATCH_SIZE = 2
SEQ_LEN = 300
IMAGE_CHANNELS = 3
IMAGE_H = IMAGE_W = 256


def make_inputs(batch=BATCH_SIZE):
    """构造测试输入"""
    x_1d = torch.randn(batch, SEQ_LEN)
    x_2d = torch.randn(batch, IMAGE_CHANNELS, IMAGE_H, IMAGE_W)
    return x_1d, x_2d


def assert_spline_head_output(out, model, batch=BATCH_SIZE):
    """验证 SplineHead 输出 dict 形状与值域"""
    assert "bbox" in out, "缺少 bbox"
    assert "keypoints" in out, "缺少 keypoints"
    assert "validity" in out, "缺少 validity"
    assert "K" in out, "缺少 K"
    assert "gp_lml" in out, "缺少 gp_lml"

    # 形状
    assert out["bbox"].shape == (batch, 4), f"bbox 应 (B, 4)，实际 {out['bbox'].shape}"
    assert out["keypoints"].shape == (batch, model.max_kpts, 2), \
        f"keypoints 应 (B, K, 2)，实际 {out['keypoints'].shape}"
    assert out["validity"].shape == (batch, model.max_kpts)
    assert out["K"].shape == (batch,)

    # 值域
    assert (out["bbox"] >= 0).all() and (out["bbox"] <= 1).all()
    assert (out["keypoints"] >= 0).all() and (out["keypoints"] <= 1).all()
    assert (out["validity"] >= 0).all() and (out["validity"] <= 1).all()
    assert (out["K"] >= model.min_kpts).all() and (out["K"] <= model.max_kpts).all()

    # bbox 顺序约束
    assert (out["bbox"][:, 0] <= out["bbox"][:, 2]).all(), "x1 <= x2"
    assert (out["bbox"][:, 1] <= out["bbox"][:, 3]).all(), "y1 <= y2"

    # GP LML 非负
    assert out["gp_lml"].item() >= 0, f"GP LML 应 >= 0，实际 {out['gp_lml'].item()}"


# ============================================================
#  PETSNetMultimodalV5 (ResNet-18)
# ============================================================

def test_resnet18_forward_shapes():
    model = PETSNetMultimodalV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    assert_spline_head_output(out, model)
    print(f"  [OK] resnet18_forward_shapes: "
          f"params={model.count_parameters():,}, K={out['K'].tolist()}")


def test_resnet18_backward():
    model = PETSNetMultimodalV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    loss = out["bbox"].sum() + out["keypoints"].sum()
    loss.backward()

    # 至少一些参数有梯度
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "至少一个参数应有梯度"
    print(f"  [OK] resnet18_backward: 梯度正常")


def test_resnet18_use_gp_false():
    kwargs = {k: v for k, v in COMMON_KWARGS.items() if k != "use_gp"}
    model = PETSNetMultimodalV5(**kwargs, use_gp=False)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    assert "gp_lml" not in out, "use_gp=False 时不应有 gp_lml"
    print(f"  [OK] resnet18_use_gp_false: 无 GP 模块")


# ============================================================
#  SwinYOLOFPNV5
# ============================================================

def test_swin_yolo_forward_shapes():
    model = SwinYOLOFPNV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    assert_spline_head_output(out, model)
    print(f"  [OK] swin_yolo_forward_shapes: "
          f"params={model.count_parameters():,}, K={out['K'].tolist()}")


def test_swin_yolo_backward():
    model = SwinYOLOFPNV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    loss = out["bbox"].sum() + out["keypoints"].sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
    print(f"  [OK] swin_yolo_backward: 梯度正常")


# ============================================================
#  ViTYOLOFPNV5
# ============================================================

def test_vit_yolo_forward_shapes():
    model = ViTYOLOFPNV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    assert_spline_head_output(out, model)
    print(f"  [OK] vit_yolo_forward_shapes: "
          f"params={model.count_parameters():,}, K={out['K'].tolist()}")


def test_vit_yolo_backward():
    model = ViTYOLOFPNV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    loss = out["bbox"].sum() + out["keypoints"].sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
    print(f"  [OK] vit_yolo_backward: 梯度正常")


# ============================================================
#  DETRStyleV5
# ============================================================

def test_detr_forward_shapes():
    model = DETRStyleV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    assert_spline_head_output(out, model)
    print(f"  [OK] detr_forward_shapes: "
          f"params={model.count_parameters():,}, K={out['K'].tolist()}")


def test_detr_backward():
    model = DETRStyleV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    loss = out["bbox"].sum() + out["keypoints"].sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
    print(f"  [OK] detr_backward: 梯度正常")


# ============================================================
#  SwinYOLOFPNWithPatchTSTV5
# ============================================================

def test_patchtst_forward_shapes():
    model = SwinYOLOFPNWithPatchTSTV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    assert_spline_head_output(out, model)
    print(f"  [OK] patchtst_forward_shapes: "
          f"params={model.count_parameters():,}, K={out['K'].tolist()}")


def test_patchtst_backward():
    model = SwinYOLOFPNWithPatchTSTV5(**COMMON_KWARGS)
    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)
    loss = out["bbox"].sum() + out["keypoints"].sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
    print(f"  [OK] patchtst_backward: 梯度正常")


# ============================================================
#  工厂函数 create_v5_model
# ============================================================

def test_factory_all_variants():
    """工厂函数应能创建所有 5 个变体"""
    variants = list(V5_MODEL_REGISTRY.keys())
    assert len(variants) == 5, f"应 5 个变体，实际 {len(variants)}"

    for variant in variants:
        model = create_v5_model(variant, **COMMON_KWARGS)
        x_1d, x_2d = make_inputs()
        out = model(x_1d, x_2d)
        assert_spline_head_output(out, model)
    print(f"  [OK] factory_all_variants: 5 个变体全部创建并 forward 成功")


def test_factory_invalid_variant():
    """无效 variant 应报错"""
    try:
        create_v5_model("invalid_variant", **COMMON_KWARGS)
        assert False, "应拒绝无效 variant"
    except ValueError:
        pass
    print(f"  [OK] factory_invalid_variant: 无效 variant 被拒绝")


# ============================================================
#  端到端：模型 + loss 集成
# ============================================================

def test_end_to_end_loss_resnet18():
    """ResNet-18 + OrderedKeypointLoss 端到端"""
    model = PETSNetMultimodalV5(**COMMON_KWARGS)
    loss_fn = OrderedKeypointLoss()

    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)

    # 构造伪 GT
    true_bbox = torch.rand(BATCH_SIZE, 4)
    true_pixels = torch.rand(BATCH_SIZE, 50, 2)

    losses = loss_fn(
        pred_bbox=out["bbox"],
        pred_kpts=out["keypoints"],
        true_bbox=true_bbox,
        true_crack_pixels=true_pixels,
        gp_module=model.gp_module,
    )
    assert losses["total"].item() > 0
    losses["total"].backward()
    print(f"  [OK] end_to_end_loss_resnet18: total={losses['total'].item():.4f}, "
          f"反向传播成功")


def test_end_to_end_loss_swin_yolo():
    """SwinYOLOFPNV5 + OrderedKeypointLoss 端到端"""
    model = SwinYOLOFPNV5(**COMMON_KWARGS)
    loss_fn = OrderedKeypointLoss()

    x_1d, x_2d = make_inputs()
    out = model(x_1d, x_2d)

    true_bbox = torch.rand(BATCH_SIZE, 4)
    true_pixels = torch.rand(BATCH_SIZE, 50, 2)

    losses = loss_fn(
        pred_bbox=out["bbox"],
        pred_kpts=out["keypoints"],
        true_bbox=true_bbox,
        true_crack_pixels=true_pixels,
        gp_module=model.gp_module,
    )
    losses["total"].backward()
    print(f"  [OK] end_to_end_loss_swin_yolo: total={losses['total'].item():.4f}")


def test_end_to_end_all_variants():
    """所有 5 个变体端到端 loss 反向"""
    loss_fn = OrderedKeypointLoss()
    true_bbox = torch.rand(BATCH_SIZE, 4)
    true_pixels = torch.rand(BATCH_SIZE, 50, 2)

    variants = list(V5_MODEL_REGISTRY.keys())
    for variant in variants:
        model = create_v5_model(variant, **COMMON_KWARGS)
        x_1d, x_2d = make_inputs()
        out = model(x_1d, x_2d)
        losses = loss_fn(
            pred_bbox=out["bbox"],
            pred_kpts=out["keypoints"],
            true_bbox=true_bbox,
            true_crack_pixels=true_pixels,
            gp_module=model.gp_module,
        )
        losses["total"].backward()
    print(f"  [OK] end_to_end_all_variants: 5 变体全部端到端反向传播通过")


# ============================================================
#  参数验证
# ============================================================

def test_invalid_image_channels():
    """无效 image_channels 应报错"""
    try:
        PETSNetMultimodalV5(image_channels=5, pretrained_2d=False)
        assert False, "应拒绝 image_channels=5"
    except ValueError:
        pass
    print(f"  [OK] invalid_image_channels: 无效通道数被拒绝")


def test_invalid_kpts():
    """无效 kpts 配置应报错"""
    try:
        PETSNetMultimodalV5(min_kpts=1, max_kpts=16, pretrained_2d=False)
        assert False, "应拒绝 min_kpts=1"
    except ValueError:
        pass

    try:
        PETSNetMultimodalV5(min_kpts=16, max_kpts=8, pretrained_2d=False)
        assert False, "应拒绝 min > max"
    except ValueError:
        pass
    print(f"  [OK] invalid_kpts: 无效关键点配置被拒绝")


# ============================================================
#  运行所有测试
# ============================================================

def run_all():
    tests = [
        # ResNet-18
        test_resnet18_forward_shapes,
        test_resnet18_backward,
        test_resnet18_use_gp_false,
        # Swin-YOLO
        test_swin_yolo_forward_shapes,
        test_swin_yolo_backward,
        # ViT-YOLO
        test_vit_yolo_forward_shapes,
        test_vit_yolo_backward,
        # DETR
        test_detr_forward_shapes,
        test_detr_backward,
        # PatchTST
        test_patchtst_forward_shapes,
        test_patchtst_backward,
        # 工厂
        test_factory_all_variants,
        test_factory_invalid_variant,
        # 端到端
        test_end_to_end_loss_resnet18,
        test_end_to_end_loss_swin_yolo,
        test_end_to_end_all_variants,
        # 参数验证
        test_invalid_image_channels,
        test_invalid_kpts,
    ]

    print("=" * 60)
    print(f"v5-α 5 变体模型单元测试（共 {len(tests)} 项）")
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

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())