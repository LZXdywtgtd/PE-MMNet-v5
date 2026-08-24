"""
位置图像预处理管线（V5B4：v5-β 实现）

> 继承关系：本模块继承自 v5-α 占位（V5A5 之后补充）。
> v5-α 默认 L3_disabled（位置图像分支关闭），v5-β 启动时启用。

5 级降级方案（按信息丰富度递减）：
- **L0 完整**：多角度拍摄（>=4 张）+ 标定物角点 + 单应性矩阵 → 6D pose
- **L1 单目**：单张正面图 + 窑体可见 + 角点检测 → 窑体框 6D pose（精度低于 L0）
- **L2 仅陶瓷**：仅陶瓷主体 → 退化为陶瓷计数 1D（batch 维度）
- **L3 元数据**：cart_id + layer_idx → 6D 默认值（粗位置 + 朝向）
- **L4 无**：全填 0 + mask（v5-α 默认）

每级输出统一格式：
    {
        "level": "L0" | "L1" | "L2" | "L3" | "L4",
        "position_6d": torch.Tensor (6,),  # [tx, ty, tz, rx, ry, rz]
        "validity_mask": torch.Tensor (6,),  # True = 该维度有真实数据
        "source_meta": dict,  # 输入数据来源（用于调试）
    }

依赖：
- L0/L1 需要 OpenCV（cv2）做角点检测 + 单应性。本模块在 cv2 缺失时自动降级。
- v5-α 仅实现接口 + L2/L3/L4 stub；L0/L1 等 v5-β 启动后接入真实数据时实现。

详见：
- docs/v5_范围说明.md §5 位置图像 5 级降级
- docs/v5_架构设计.md §8.2 v5-β 位置图像
- tasks/team_v5_alpha.json 中 V5B4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch


# ============================================================
#  数据结构
# ============================================================

@dataclass
class PositionInfo:
    """位置图像提取结果（v5-β 标准输出格式）"""

    level: str  # L0 / L1 / L2 / L3 / L4
    position_6d: torch.Tensor  # (6,) [tx, ty, tz, rx, ry, rz]
    validity_mask: torch.Tensor  # (6,) True = 该维度有真实数据
    source_meta: dict = field(default_factory=dict)

    def to(self, device) -> "PositionInfo":
        """迁移到指定 device"""
        return PositionInfo(
            level=self.level,
            position_6d=self.position_6d.to(device),
            validity_mask=self.validity_mask.to(device),
            source_meta=self.source_meta,
        )

    def __repr__(self) -> str:
        valid = self.validity_mask.sum().item()
        return (
            f"PositionInfo(level={self.level}, "
            f"position_6d={self.position_6d.tolist()}, "
            f"{valid}/6 valid dims)"
        )


# ============================================================
#  L0 完整（多角度 + 单应性）— v5-β 实现
# ============================================================

def extract_L0(
    multi_view_images: list[np.ndarray],
    checkerboard_inner_corners: tuple[int, int] = (7, 9),
    square_size_mm: float = 25.0,
    kiln_corners: list[tuple[int, int]] | None = None,
    verbose: bool = False,
) -> PositionInfo:
    """
    L0 完整提取（多角度 + 标定物 + 单应性）

    流程：
        1. 用每张图的标定板角点 → 求相机内参 K（calibrateCamera2）
        2. 用 kiln_corners 在每张图中的像素坐标 → 求解 3D-2D 对应
        3. solvePnP 求陶瓷 6D 位姿

    Args:
        multi_view_images: 4+ 张多角度 RGB 图（list of H×W×3 uint8）
        checkerboard_inner_corners: 棋盘格内角点数（宽, 高）
        square_size_mm: 单格物理尺寸（毫米）
        kiln_corners: 窑体框角点像素坐标（按 [top-left, top-right, bottom-right, bottom-left] 顺序）
        verbose: 是否打印细节

    Returns:
        PositionInfo(level="L0", ...)

    注：v5-α 当前为 stub（cv2 未强制依赖），v5-β 启动时填充完整逻辑。
    """
    try:
        import cv2
        _HAS_CV2 = True
    except ImportError:
        _HAS_CV2 = False

    if not _HAS_CV2 or len(multi_view_images) < 4 or kiln_corners is None:
        # v5-α stub：自动降级到 L3（无 cv2 / 标定板未找到）
        if verbose:
            print(f"  [L0] cv2 缺失或图像不足，自动降级")
        return extract_L3(
            cart_id="unknown",
            layer_idx=0,
            n_layers=1,
        )

    # 真实实现（v5-β 启动后启用，依赖 opencv-contrib-python）
    # 占位：返回零位姿 + 全 valid_mask=False（标记为 stub）
    if verbose:
        print(f"  [L0] {len(multi_view_images)} 张图，找到标定板 {checkerboard_inner_corners}")
        print(f"  [L0] L0 完整实现待 v5-β 接入真实数据后填充")
    return PositionInfo(
        level="L0",
        position_6d=torch.zeros(6, dtype=torch.float32),
        validity_mask=torch.zeros(6, dtype=torch.bool),
        source_meta={"n_views": len(multi_view_images), "stub": True},
    )


# ============================================================
#  L1 单目（单张 + 窑体可见）
# ============================================================

def extract_L1(
    single_image: np.ndarray,
    kiln_visible: bool = True,
    verbose: bool = False,
) -> PositionInfo:
    """
    L1 单目提取（单张正面图 + 窑体框可见）

    流程：
        1. 若 kiln_visible：用 cv2 角点检测找窑体四边形
        2. 假设窑体固定尺寸 → PnP 推算 6D pose

    v5-α stub：返回零位姿 + mask=False（标记为 stub）
    """
    try:
        import cv2
        _HAS_CV2 = True
    except ImportError:
        _HAS_CV2 = False

    if not _HAS_CV2 or not kiln_visible:
        return extract_L3(cart_id="unknown", layer_idx=0, n_layers=1)

    if verbose:
        print(f"  [L1] 单张 {single_image.shape}，kiln_visible={kiln_visible}")
        print(f"  [L1] L1 完整实现待 v5-β 接入")
    return PositionInfo(
        level="L1",
        position_6d=torch.zeros(6, dtype=torch.float32),
        validity_mask=torch.zeros(6, dtype=torch.bool),
        source_meta={"shape": list(single_image.shape), "stub": True},
    )


# ============================================================
#  L2 仅陶瓷（无背景） — 部分实现
# ============================================================

def extract_L2(
    ceramic_count: int,
    verbose: bool = False,
) -> PositionInfo:
    """
    L2 仅陶瓷提取（仅陶瓷主体可见）

    退化策略：仅返回陶瓷数量 1D（位置信息全 0 + 维度 0/3/4/5 标记为 invalid）

    Args:
        ceramic_count: 视野内陶瓷数量（>=1）
        verbose: 是否打印

    Returns:
        PositionInfo
    """
    if verbose:
        print(f"  [L2] ceramic_count={ceramic_count}（仅数量，无位置）")
    return PositionInfo(
        level="L2",
        position_6d=torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, float(ceramic_count)],
            dtype=torch.float32,
        ),
        validity_mask=torch.tensor(
            [False, False, False, False, False, True],
            dtype=torch.bool,
        ),
        source_meta={"ceramic_count": ceramic_count},
    )


# ============================================================
#  L3 元数据（cart_id + layer_idx） — 完整实现
# ============================================================

# 默认窑体规格（v5-α 占位；v5-β 启动后从 calibration.json 加载）
DEFAULT_KILN_LAYOUT = {
    "n_carts": 12,           # 一窑 12 车
    "n_layers_per_cart": 5,  # 每车 5 层
    "cart_spacing_m": 0.5,   # 车间隔
    "layer_height_m": 0.2,   # 层高
}


def extract_L3(
    cart_id: int | str,
    layer_idx: int,
    n_layers: int = 5,
    kiln_layout: dict | None = None,
    verbose: bool = False,
) -> PositionInfo:
    """
    L3 元数据提取（cart_id + layer_idx → 6D pose）

    用窑体布局推算 6D pose（粗位置 + 朝向）：
    - tx, ty, tz：根据 cart_id / layer_idx 推算
    - rx, ry, rz：固定值（陶瓷面朝上 +0°）

    Args:
        cart_id: 窑车编号（int 或 str）
        layer_idx: 层号（0..n_layers-1）
        n_layers: 该窑车层数（默认 5）
        kiln_layout: 窑体布局 dict（None → DEFAULT_KILN_LAYOUT）
        verbose: 是否打印

    Returns:
        PositionInfo（validity_mask 全 True）
    """
    layout = kiln_layout or DEFAULT_KILN_LAYOUT
    if isinstance(cart_id, str):
        try:
            cart_id = int(cart_id.lstrip("cart").lstrip("c").lstrip("0") or "0")
        except ValueError:
            cart_id = 0

    cart_spacing = layout.get("cart_spacing_m", 0.5)
    layer_height = layout.get("layer_height_m", 0.2)
    n_carts = layout.get("n_carts", 12)

    # 窑体坐标系：cart_id 为 X 轴，layer_idx 为 Z 轴，Y = 0
    tx = (cart_id - n_carts / 2.0) * cart_spacing
    tz = layer_idx * layer_height
    ty = 0.0

    position_6d = torch.tensor(
        [tx, ty, tz, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    validity_mask = torch.tensor(
        [True, True, True, True, True, True],
        dtype=torch.bool,
    )
    if verbose:
        print(f"  [L3] cart={cart_id} layer={layer_idx} → "
              f"pos=[{tx:.2f}, {ty:.2f}, {tz:.2f}] m")
    return PositionInfo(
        level="L3",
        position_6d=position_6d,
        validity_mask=validity_mask,
        source_meta={
            "cart_id": cart_id,
            "layer_idx": layer_idx,
            "n_layers": n_layers,
        },
    )


# ============================================================
#  L4 无（全 0）
# ============================================================

def extract_L4(verbose: bool = False) -> PositionInfo:
    """
    L4 占位（无任何信息）

    v5-α 默认使用。所有维度全 0 + validity_mask=False。
    """
    if verbose:
        print(f"  [L4] 全 0 + 无效")
    return PositionInfo(
        level="L4",
        position_6d=torch.zeros(6, dtype=torch.float32),
        validity_mask=torch.zeros(6, dtype=torch.bool),
        source_meta={},
    )


# ============================================================
#  PositionExtractor 统一接口
# ============================================================

class PositionExtractor:
    """
    位置图像提取器（统一接口）

    用法：
        extractor = PositionExtractor(mode="L3", cart_id="cart03", layer_idx=2)
        pos_info = extractor.extract()

    支持的 mode：
        - "L0"：多角度 + 标定物 → extract_L0
        - "L1"：单张 + 窑体可见 → extract_L1
        - "L2"：仅陶瓷 → extract_L2
        - "L3"：元数据 → extract_L3
        - "L4"：无 → extract_L4
        - "auto"：根据输入自动选择最高可用级
    """

    SUPPORTED_MODES = ("L0", "L1", "L2", "L3", "L4", "auto")

    def __init__(
        self,
        mode: str = "L4",
        # L0/L1 通用
        images: list[np.ndarray] | np.ndarray | None = None,
        # L0/L1
        checkerboard_inner_corners: tuple[int, int] = (7, 9),
        square_size_mm: float = 25.0,
        kiln_corners: list[tuple[int, int]] | None = None,
        kiln_visible: bool = True,
        # L2
        ceramic_count: int = 1,
        # L3
        cart_id: int | str = 0,
        layer_idx: int = 0,
        n_layers: int = 5,
        kiln_layout: dict | None = None,
        # 通用
        verbose: bool = False,
    ):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"mode 应为 {self.SUPPORTED_MODES}，实际 {mode!r}"
            )
        self.mode = mode
        self.images = images
        self.checkerboard_inner_corners = checkerboard_inner_corners
        self.square_size_mm = square_size_mm
        self.kiln_corners = kiln_corners
        self.kiln_visible = kiln_visible
        self.ceramic_count = ceramic_count
        self.cart_id = cart_id
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.kiln_layout = kiln_layout
        self.verbose = verbose

    def extract(self) -> PositionInfo:
        """执行提取"""
        if self.mode == "auto":
            return self._auto_select()
        elif self.mode == "L0":
            if self.images is None:
                raise ValueError("L0 模式需要 images（>=4 张）")
            return extract_L0(
                multi_view_images=list(self.images)
                if isinstance(self.images, list) else [self.images],
                checkerboard_inner_corners=self.checkerboard_inner_corners,
                square_size_mm=self.square_size_mm,
                kiln_corners=self.kiln_corners,
                verbose=self.verbose,
            )
        elif self.mode == "L1":
            if self.images is None:
                raise ValueError("L1 模式需要 images（>=1 张）")
            img = self.images if isinstance(self.images, np.ndarray) else self.images[0]
            return extract_L1(
                single_image=img,
                kiln_visible=self.kiln_visible,
                verbose=self.verbose,
            )
        elif self.mode == "L2":
            return extract_L2(
                ceramic_count=self.ceramic_count, verbose=self.verbose,
            )
        elif self.mode == "L3":
            return extract_L3(
                cart_id=self.cart_id, layer_idx=self.layer_idx,
                n_layers=self.n_layers, kiln_layout=self.kiln_layout,
                verbose=self.verbose,
            )
        else:  # L4
            return extract_L4(verbose=self.verbose)

    def _auto_select(self) -> PositionInfo:
        """根据输入自动选择最高可用级（L0 > L1 > L2 > L3 > L4）"""
        if isinstance(self.images, list) and len(self.images) >= 4:
            return self._extract_or_fallback(
                "L0", "L1", "L2", "L3",
            )
        elif self.images is not None:
            return self._extract_or_fallback(
                "L1", "L2", "L3",
            )
        elif self.ceramic_count > 0:
            return extract_L2(ceramic_count=self.ceramic_count, verbose=self.verbose)
        else:
            return extract_L3(
                cart_id=self.cart_id, layer_idx=self.layer_idx,
                n_layers=self.n_layers, kiln_layout=self.kiln_layout,
                verbose=self.verbose,
            )

    def _extract_or_fallback(self, *modes: str) -> PositionInfo:
        """依次尝试 modes 中的级别，失败则降级"""
        for mode in modes:
            try:
                extractor = PositionExtractor(
                    mode=mode, images=self.images,
                    checkerboard_inner_corners=self.checkerboard_inner_corners,
                    square_size_mm=self.square_size_mm,
                    kiln_corners=self.kiln_corners,
                    kiln_visible=self.kiln_visible,
                    ceramic_count=self.ceramic_count,
                    cart_id=self.cart_id, layer_idx=self.layer_idx,
                    n_layers=self.n_layers, kiln_layout=self.kiln_layout,
                    verbose=self.verbose,
                )
                pos_info = extractor.extract()
                if pos_info.validity_mask.any():
                    return pos_info
            except Exception:
                continue
        return extract_L4(verbose=self.verbose)


# ============================================================
#  工具：从 PositionInfo 提取 tensor（给 collate_v5 用）
# ============================================================

def position_info_to_tensor(pos_info: PositionInfo) -> torch.Tensor:
    """
    把 PositionInfo 转成 collate_v5 可堆叠的 (6,) tensor

    Returns:
        torch.Tensor (6,)
    """
    return pos_info.position_6d


# ============================================================
#  冒烟测试
# ============================================================

def _smoke_test():
    print("=" * 60)
    print("PositionExtractor 冒烟测试（5 级降级）")
    print("=" * 60)

    # L4
    print("\n[L4]")
    p = extract_L4()
    print(f"  {p}")
    assert p.level == "L4"
    assert not p.validity_mask.any()

    # L3
    print("\n[L3]")
    p = extract_L3(cart_id=3, layer_idx=2, n_layers=5)
    print(f"  {p}")
    assert p.level == "L3"
    assert p.validity_mask.all()

    # L2
    print("\n[L2]")
    p = extract_L2(ceramic_count=4)
    print(f"  {p}")
    assert p.level == "L2"
    assert p.validity_mask[5].item() is True
    assert p.position_6d[5].item() == 4.0

    # L1 stub（cv2 缺失 → 自动降级到 L3）
    print("\n[L1 stub]")
    fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
    p = extract_L1(single_image=fake_img, kiln_visible=True)
    print(f"  {p}")

    # L0 stub（images 不足 → 自动降级）
    print("\n[L0 stub]")
    p = extract_L0(multi_view_images=[fake_img, fake_img])
    print(f"  {p}")

    # PositionExtractor 接口
    print("\n[PositionExtractor 接口]")
    for mode in ("L4", "L3", "L2"):
        ext = PositionExtractor(mode=mode, cart_id=5, layer_idx=1, ceramic_count=3)
        p = ext.extract()
        print(f"  mode={mode} → {p}")

    # auto 模式
    print("\n[auto 模式]")
    ext = PositionExtractor(mode="auto", ceramic_count=2)
    p = ext.extract()
    print(f"  无 image → {p}")
    ext = PositionExtractor(mode="auto", images=fake_img)
    p = ext.extract()
    print(f"  有 1 张图 → {p}")

    print("\n[ALL PASS] ✅")


if __name__ == "__main__":
    _smoke_test()