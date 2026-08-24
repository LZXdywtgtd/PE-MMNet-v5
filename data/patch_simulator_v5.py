"""
patch 仿真器（v5-α 占位）

> 待 V5A1 任务实现。本文件作为 v5-α 占位，提醒后续开发者这里需要：
>   1. S1 多方向热交换（FDM 替换 v4 的简化积分）
>   2. S2 表面辐射（Stefan-Boltzmann）
>   3. S3 内部应力松弛（Maxwell 粘弹性）
>   4. S4 三段升降温曲线
>   5. S7 像素级裂纹 mask 输出 + 关键点 GT 提取

详见：
- docs/v5_仿真升级清单.md
- tasks/team_v5_alpha.json 中 V5A1
"""

import numpy as np


class PatchSimulator:
    """256×256 patch 仿真器（v5-α 占位）"""

    def __init__(
        self,
        patch_size: int = 256,
        thermal_profile: dict | None = None,
        boundary_config: dict | None = None,
        enable_growth: bool = False,  # 是否启用 S5/S6（Griffith + LEFM）
    ):
        self.patch_size = patch_size
        self.thermal_profile = thermal_profile
        self.boundary_config = boundary_config
        self.enable_growth = enable_growth

        # TODO(V5A1): 实现仿真器
        raise NotImplementedError(
            "PatchSimulator 待 V5A1 实现。详见 docs/v5_仿真升级清单.md"
        )

    def simulate(self, seed: int = 42) -> dict:
        """
        仿真单次烧制过程

        Returns:
            dict containing:
                temperature_field: (T_steps, 256, 256)
                stress_field:      (T_steps, 256, 256)
                heatmap:           (256, 256)
                crack_mask:        (256, 256)
                crack_keypoints:   (K, 2), K ∈ [8, 16]
                crack_bbox:        (4,)
        """
        raise NotImplementedError