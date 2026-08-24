"""
PE-MMNet v4 模型模块

导出所有可用的模型类
"""

# 原有模型
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
