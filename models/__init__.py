"""
PE-MMNet 模型模块

导出 v4 (默认) 和 v5-α 模型：
- v4: 6 维向量输出 [x, y, l, w, conf, density]
- v5: bbox + 8-16 动态关键点 + GP 物理约束
"""

# ============================================================
#  v4 模型（默认）
# ============================================================

from .pe_tsnet_multimodal import (
    PETSNetMultimodal,
    PETSNetMultimodalSmall,
    ConfigurableMultimodal,
    create_model,
    get_arch_specific_config,
    # 骨干网络
    ResNet18Backbone2D,
    ViTBackbone2D,
    # 1D 编码器
    TemporalFeatureExtractor,
    DLinear1D,
    Transformer1D,
    # 融合层
    CrossAttentionFusion,
    AdaptiveFusion,
    # 输出头
    MultiTaskHead,
    MaskDecoder,
)

# YOLO-FPN 变体
from .pe_tsnet_yolo import (
    SwinYOLOFPN,
    ViTYOLOFPN,
    SwinBackbone2D,
    ViTYOLOBackbone2D,
    FPNNeck,
    YOLOFPNHead,
)

# DETR 风格变体
from .pe_tsnet_detr import (
    DETRStyle,
    DETRDecoder,
    DETRHead,
    PositionalEncoding2D,
)

# PatchTST 变体
from .pe_tsnet_patchtst import (
    PatchTST1D,
    PatchTSTWithRate,
)

# ============================================================
#  v5-α 模型（V5A4 新增）
# ============================================================

from .spline_head import SplineHead
from .gp_module import GPModule

from .pe_tsnet_multimodal_v5 import (
    PETSNetMultimodalV5,
)

from .pe_tsnet_yolo_v5 import (
    SwinYOLOFPNV5,
    ViTYOLOFPNV5,
    ViTYOLOBackbone2DV5,
)

from .pe_tsnet_detr_v5 import (
    DETRStyleV5,
)

from .pe_tsnet_patchtst_v5 import (
    SwinYOLOFPNWithPatchTSTV5,
)


# ============================================================
#  v5-α 模型注册表（用于统一入口）
# ============================================================

V5_MODEL_REGISTRY = {
    "resnet18": {
        "class": PETSNetMultimodalV5,
        "module": "models.pe_tsnet_multimodal_v5",
        "desc": "ResNet-18 + CrossAttentionFusion + SplineHead + GP",
    },
    "swin_yolo": {
        "class": SwinYOLOFPNV5,
        "module": "models.pe_tsnet_yolo_v5",
        "desc": "Swin-Tiny + YOLO-FPN + TemporalFeatureExtractor + SplineHead + GP",
    },
    "vit_yolo": {
        "class": ViTYOLOFPNV5,
        "module": "models.pe_tsnet_yolo_v5",
        "desc": "ViT-Small + YOLO-FPN + TemporalFeatureExtractor + SplineHead + GP",
    },
    "detr": {
        "class": DETRStyleV5,
        "module": "models.pe_tsnet_detr_v5",
        "desc": "ResNet-18 + Transformer Encoder-Decoder + SplineHead + GP",
    },
    "swin_yolo_patchtst": {
        "class": SwinYOLOFPNWithPatchTSTV5,
        "module": "models.pe_tsnet_patchtst_v5",
        "desc": "Swin-Tiny + YOLO-FPN + PatchTST 1D + SplineHead + GP",
    },
}


def create_v5_model(variant: str, **kwargs):
    """
    工厂函数：按 variant 名称创建 v5-α 模型

    Args:
        variant: "resnet18" | "swin_yolo" | "vit_yolo" | "detr" | "swin_yolo_patchtst"
        **kwargs: 传给对应模型类的参数

    Returns:
        nn.Module 实例

    Raises:
        ValueError: 如果 variant 未知
    """
    if variant not in V5_MODEL_REGISTRY:
        valid = ", ".join(V5_MODEL_REGISTRY.keys())
        raise ValueError(f"未知 variant {variant!r}，可选: {valid}")

    entry = V5_MODEL_REGISTRY[variant]
    return entry["class"](**kwargs)


__all__ = [
    # v4
    "PETSNetMultimodal", "PETSNetMultimodalSmall", "ConfigurableMultimodal",
    "create_model", "get_arch_specific_config",
    "ResNet18Backbone2D", "ViTBackbone2D",
    "TemporalFeatureExtractor", "DLinear1D", "Transformer1D",
    "CrossAttentionFusion", "AdaptiveFusion",
    "MultiTaskHead", "MaskDecoder",
    "SwinYOLOFPN", "ViTYOLOFPN", "SwinBackbone2D",
    "ViTYOLOBackbone2D", "FPNNeck", "YOLOFPNHead",
    "DETRStyle", "DETRDecoder", "DETRHead", "PositionalEncoding2D",
    "PatchTST1D", "PatchTSTWithRate",
    # v5-α
    "SplineHead", "GPModule",
    "PETSNetMultimodalV5",
    "SwinYOLOFPNV5", "ViTYOLOFPNV5", "ViTYOLOBackbone2DV5",
    "DETRStyleV5",
    "SwinYOLOFPNWithPatchTSTV5",
    "V5_MODEL_REGISTRY",
    "create_v5_model",
]