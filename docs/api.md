# API 参考文档

> PE-MMNet v5-α 公开 API | 版本 5.0.5-alpha | 更新：2026-08-24
>
> **继承说明**：本文件由 v4.6.11 同名文档**完全重写**。v5-α **抛弃了** v4 的 6 维向量 / 二值掩膜 / `train_model` / `team_train.py` 等 API；v5-α 公开 API 全部为新写。
>
> 所有签名通过 `inspect.signature()` 在 commit `<latest>` 验证。

---

## 目录

| § | 模块 | 说明 |
|---|------|------|
| §0 | v5-α 完整 API 索引 | 所有公开入口 |
| §1 | 模型（`models/`）| 5 变体工厂 + SplineHead + GPModule |
| §2 | 数据（`data/`）| PatchSimulator + PatchDatasetV5 + collate_v5 |
| §3 | 训练（`training/`）| TrainerV5 + OrderedKeypointLoss + PatchAugmentorV5 |
| §4 | 工具（`tools/`）| 可视化 + 检查 |
| §5 | 启动器（`run_train_v5.py`）| CLI 入口 |

---

## §0. v5-α 完整 API 索引

```python
# 模型
from models import create_v5_model, V5_MODEL_REGISTRY
from models import SplineHead, GPModule
from models.pe_tsnet_multimodal_v5 import PETSNetMultimodalV5
from models.pe_tsnet_yolo_v5 import SwinYOLOFPNV5, ViTYOLOFPNV5, ViTYOLOBackbone2DV5
from models.pe_tsnet_detr_v5 import DETRStyleV5
from models.pe_tsnet_patchtst_v5 import SwinYOLOFPNWithPatchTSTV5

# 数据
from data.patch_simulator_v5 import PatchSimulator
from data.patch_dataset_v5 import PatchDatasetV5, collate_v5
from data.thermal_profile import generate_thermal_profile, total_duration_s
from data.surface_radiation import SurfaceRadiation
from data.viscoelastic import ViscoelasticStress

# 训练
from training.trainer_v5 import (
    TrainerV5, CSVHistory, save_checkpoint, load_checkpoint,
    set_seed, get_device,
)
from training.ordered_kp_loss import OrderedKeypointLoss
from training.data_aug_v5 import PatchAugmentorV5

# 工具
from tools.patch_simulator_visualizer import visualize_patch
```

---

## §1. 模型（`models/`）

### 1.1 工厂函数

```python
def create_v5_model(
    variant: str,
    image_channels: int = 3,
    pretrained_2d: bool = False,
    use_gp: bool = True,
    min_kpts: int = 8,
    max_kpts: int = 16,
    **kwargs,
) -> nn.Module
```

**说明**：v5-α 唯一的模型创建入口。5 变体（`resnet18` / `swin_yolo` / `vit_yolo` / `detr` / `swin_yolo_patchtst`）通过此工厂创建。

| 参数 | 默认 | 说明 |
|------|------|------|
| `variant` | - | 5 变体 key 之一 |
| `image_channels` | 3 | 输入通道数（v5-α 固定 3） |
| `pretrained_2d` | False | 是否加载 ImageNet 预训练 |
| `use_gp` | True | 是否启用 GP 嵌入模块 |
| `min_kpts`, `max_kpts` | 8, 16 | 关键点动态范围 |

**示例**：

```python
from models import create_v5_model
import torch

model = create_v5_model("resnet18", image_channels=3, use_gp=True, min_kpts=8, max_kpts=16)
model = model.cuda()
x_1d = torch.randn(4, 300).cuda()         # (B, seq_len) 1D 时序
x_2d = torch.randn(4, 3, 256, 256).cuda() # (B, C, H, W) 2D patch
out = model(x_1d, x_2d)
# out: dict
#   bbox:        (B, 4) sigmoid-normalized [0, 1]
#   keypoints:   (B, max_kpts, 2) 关键点 (y, x)
#   validity:    (B, max_kpts) bool mask（动态 K）
#   K:           (B,) int，动态关键点数 ∈ [min_kpts, max_kpts]
#   gp_lml:      scalar（仅 use_gp=True）
```

### 1.2 SplineHead（v5-α 新增）

```python
class SplineHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        min_kpts: int = 8,
        max_kpts: int = 16,
        hidden_dim: int = 256,
        predict_kpts_only: bool = False,
    ):
        """
        Spline 输出头：
        - bbox: (B, 4) sigmoid-normalized [0, 1]（x1≤x2, y1≤y2 re-param）
        - keypoints: (B, max_kpts, 2) 浮点 (y, x)
        - validity: (B, max_kpts) bool mask（动态 K）
        - K: (B,) int
        """
```

**辅助方法**：

```python
def predict_kpts_only(model, sort_by: str = "x") -> torch.Tensor:
    """
    仅返回关键点（按 sort_by 排序）
    sort_by: "x" | "validity"
    """
```

### 1.3 GPModule（v5-α 新增）

```python
class GPModule(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        learnable: bool = True,
        init_log_lengthscale: float = 0.0,
        init_log_variance: float = 0.0,
    ):
        """
        高斯过程嵌入模块：
        - RBF kernel: k(x, x') = σ² exp(-||x-x'||² / (2ℓ²))
        - L2 row-normalize by sqrt(in_dim) 后再算 kernel（V5-014/015 记录）
        - log_marginal_likelihood 用作正则
        """
```

---

## §2. 数据（`data/`）

### 2.1 PatchSimulator（V5A1）

```python
class PatchSimulator:
    def __init__(
        self,
        patch_size: int = 256,
        physical_size_cm: float = 5.0,
        thermal_profile: dict | None = None,
        boundary_config: dict | None = None,
        material: dict | None = None,
        dt_s: float | None = None,
        crack_stress_threshold_MPa: float = 50.0,
        min_keypoints: int = 8,
        max_keypoints: int = 16,
        seed: int = 42,
    ):
        """
        256×256 patch 仿真器
        - S1: 多方向热交换（FDM 显式 Euler）
        - S2: 表面辐射（Stefan-Boltzmann）
        - S3: 内部应力松弛（Maxwell 粘弹性）
        - S4: 三段升降温曲线
        - S7: 像素级裂纹 mask 输出
        """
```

```python
def simulate(
    self,
    sample_interval_steps: int = 60,
    verbose: bool = False,
) -> dict:
    """
    Returns:
        temperature_field:    (T_samples, H, W)
        stress_field:         (T_samples, H, W)
        heatmap:              (H, W)
        crack_mask:           (H, W) uint8
        crack_keypoints:      (K, 2), K ∈ [8, 16]
        crack_bbox:           (4,)
        metadata:             dict
    """
```

### 2.2 PatchDatasetV5 + collate_v5（V5A5）

```python
class PatchDatasetV5(Dataset):
    def __init__(
        self,
        patch_size: int = 256,
        n_samples: int = 100,
        min_kpts: int = 8,
        max_kpts: int = 16,
        seed: int = 42,
        thermal_profile: dict | None = None,
        crack_stress_threshold_MPa: float = 50.0,
        precompute: bool = True,
    ):
        """
        数据集：__init__ 时一次性仿真所有样本到内存 cache
        每个 256×256 样本 ~2MB → 1000 样本 ~2GB（V5-016）
        """
```

```python
def collate_v5(batch: list[dict]) -> dict:
    """
    变长批处理：把不同长度的 true_keypoints / true_crack_pixels
    padding 到 batch 内 max_K / max_N + mask 标记
    
    Returns dict:
        x_1d:              (B, seq_len)
        x_2d:              (B, 3, H, W)
        true_bbox:         (B, 4)
        true_keypoints:    (B, max_K, 2) (-1 标记 padding)
        keypoint_mask:     (B, max_K) bool
        true_crack_pixels: (B, max_N, 2) (-1 标记 padding)
        pixel_mask:        (B, max_N) bool
        metadata:          list[dict]
    """
```

### 2.3 物理子模块

```python
from data.thermal_profile import generate_thermal_profile, total_duration_s

def generate_thermal_profile(
    ramp_up_c_per_min: float = 5.0,
    soak_temp_c: float = 1280.0,
    soak_duration_min: float = 30.0,
    cool_down_c_per_min: float = 3.0,
    room_temp_c: float = 20.0,
    dt_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成三段升降温曲线
    Returns (times_s, target_temps_c)
    """
```

```python
from data.surface_radiation import SurfaceRadiation
from data.viscoelastic import ViscoelasticStress

class SurfaceRadiation:
    """Stefan-Boltzmann 辐射"""
class ViscoelasticStress:
    """Maxwell 粘弹性应力更新"""
```

---

## §3. 训练（`training/`）

### 3.1 TrainerV5（V5A5）

```python
class TrainerV5:
    def __init__(
        self,
        variant: str = "resnet18",
        n_samples: int = 100,
        patch_size: int = 256,
        batch_size: int = 4,
        epochs: int = 10,
        lr: float = 1e-4,
        min_kpts: int = 8,
        max_kpts: int = 16,
        coverage_mode: str = "chamfer",   # "chamfer" | "hausdorff"
        M: int = 200,                      # 样条采样点数
        lambda_coverage: float = 1.0,
        lambda_bbox: float = 1.0,
        lambda_gp: float = 0.1,
        lambda_poisson: float = 0.05,
        use_gp: bool = True,
        use_aug: bool = True,
        seed: int = 42,
        device: str | None = None,
        log_dir: str = "logs/training_history/run_train_v5",
        thermal_profile: dict | None = None,
        crack_stress_threshold_MPa: float = 50.0,
        save_every: int = 1,
        model_kwargs: dict | None = None,
    ):
        """
        训练器：单任务、单 GPU
        """
```

```python
def fit(
    self,
    resume_from: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Returns:
        {
            "last_train_loss": float,
            "last_val_loss":   float,
            "best_val_loss":   float,
        }
    """
```

### 3.2 CSVHistory + 检查点

```python
class CSVHistory:
    """每个 epoch 写一行到 training_history.csv"""

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    extra: dict | None = None,
) -> None:
    """保存 best.pt / latest.pt（含 RNG 状态）"""

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """加载检查点并恢复 RNG 状态"""

def set_seed(seed: int) -> None:
    """python / numpy / torch 三种 RNG 同时播种"""

def get_device(prefer_cuda: bool = True) -> torch.device:
    """自动选 cuda / cpu"""
```

### 3.3 OrderedKeypointLoss（V5A3）

```python
class OrderedKeypointLoss(nn.Module):
    def __init__(
        self,
        min_kpts: int = 8,
        max_kpts: int = 16,
        coverage_mode: str = "chamfer",   # "chamfer" | "hausdorff"
        M: int = 200,
        lambda_coverage: float = 1.0,
        lambda_bbox: float = 1.0,
        lambda_gp: float = 0.1,
        lambda_poisson: float = 0.05,
        gp_module: GPModule | None = None,
    ):
        """
        损失 = coverage + bbox + gp + poisson
        Catmull-Rom 样条拟合预测关键点 → 等弧长采样 M 个点
        Chamfer/Hausdorff 距离到真值裂纹像素
        """
```

```python
def forward(
    self,
    pred_bbox: torch.Tensor,           # (B, 4)
    pred_kpts: torch.Tensor,           # (B, max_K, 2)
    kpt_validity: torch.Tensor,        # (B, max_K) bool
    true_bbox: torch.Tensor,           # (B, 4)
    true_crack_pixels: torch.Tensor,   # (B, N, 2) 或 (B, 0, 2)（空）
    pixel_mask: torch.Tensor | None = None,  # (B, N) bool
) -> dict:
    """
    Returns dict:
        total: scalar
        coverage: scalar
        bbox: scalar
        gp: scalar (None if no gp_module)
        poisson: scalar
    """
```

### 3.4 PatchAugmentorV5（V5A5）

```python
class PatchAugmentorV5:
    """
    物理安全增强：hflip + vflip + rot90 + Gaussian noise
    bbox / 关键点 / 像素坐标随图像同步变换
    bbox 顺序（x1≤x2, y1≤y2）、值域 [0,1] 在增强后仍成立
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        gaussian_noise_std: float = 0.01,
    ):
        ...

    def __call__(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor,
        bbox: torch.Tensor,
        kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        pixels: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> tuple:
        ...
```

---

## §4. 工具（`tools/`）

### 4.1 可视化

```python
from tools.patch_simulator_visualizer import visualize_patch

def visualize_patch(
    temperature_field: np.ndarray,  # (T, H, W) 或 (H, W)
    stress_field: np.ndarray,        # 同上
    crack_mask: np.ndarray,         # (H, W) uint8
    keypoints: np.ndarray,          # (K, 2)
    bbox: np.ndarray,               # (4,)
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    物理因果链展示图（V5A6 用）：
    温度场 + 应力场 + 裂纹 mask + 关键点轨迹 的 2x2 网格图
    """
```

---

## §5. 启动器（`run_train_v5.py`）

v5-α 唯一的 CLI 入口。完整 CLI 参数：

```bash
python run_train_v5.py \
    --variant {resnet18|swin_yolo|vit_yolo|detr|swin_yolo_patchtst} \
    --epochs N \
    --n_samples N \
    --patch_size 256 \
    --batch_size 4 \
    --lr 1e-4 \
    --min_kpts 8 \
    --max_kpts 16 \
    --coverage_mode {chamfer|hausdorff} \
    --M 200 \
    --lambda_coverage 1.0 \
    --lambda_bbox 1.0 \
    --lambda_gp 0.1 \
    --lambda_poisson 0.05 \
    --save_every 1 \
    --device {cuda|cpu|None} \
    --log_dir logs/training_history/<run> \
    --seed 42 \
    --resume <checkpoint.pt> \
    --fast_thermal \
    --no_gp \
    --no_aug \
    --pretrained_2d \
    --crack_stress_threshold_MPa 50.0
```

详细说明见 [快速配置指南.md §五](快速配置指南.md)。

---

## §6. 验证签名

```bash
/c/Users/LZXdywtgtd/.conda/envs/pe_mmnet/python.exe -c "
import inspect
from training.trainer_v5 import TrainerV5
from models import create_v5_model
from data.patch_simulator_v5 import PatchSimulator
from data.patch_dataset_v5 import collate_v5
from training.ordered_kp_loss import OrderedKeypointLoss

print('TrainerV5.__init__:', list(inspect.signature(TrainerV5.__init__).parameters))
print('create_v5_model:   ', list(inspect.signature(create_v5_model).parameters))
print('PatchSimulator:    ', list(inspect.signature(PatchSimulator.__init__).parameters))
print('collate_v5:        ', list(inspect.signature(collate_v5).parameters))
print('OrderedKeypointLoss:', list(inspect.signature(OrderedKeypointLoss.__init__).parameters))
"
```

---

## §7. 本文件的历史

- **2026-08-12（V5A5）**：fork 自 v4.6.11 同名文档，§0 加 v5-α 标注
- **2026-08-24（本次）**：完全废弃 v4 内容（`train_model` / `evaluate_model` / `team_train.py` / `freeze_model_backbone` 等），改为 v5-α 完整 API 索引

---

## §8. v5-β 规划 API（待实现）

> **触发条件**：V5A6 评估通过。
> **目标**：把 v5-α 算法骨架接入真实 3D 数据。
> **预估工作量**：~16 周（4 个月）

### 8.1 V5B1 — UV 展开管线

```python
# tools/uv_unwrap.py
class UVUnwrapper:
    """
    3D mesh → 2D UV 映射
    - 凸形状：xatlas 自动展开
    - 凹形状：手工分割（杯口、碗底等）
    """
    def __init__(self, mesh_path: str, target_size: int = 256):
        ...

    def unwrap(self) -> UVAtlas:
        ...

# data/uv_mapper.py
class UVMapper:
    """
    UV atlas ↔ 3D 表面坐标
    """
    def patch_to_3d(self, y: int, x: int) -> tuple[float, float, float]:
        ...

    def crack_3d_to_patch(self, crack_3d: np.ndarray) -> np.ndarray:
        """真实裂纹 3D 坐标 → 2D patch 像素坐标"""
        ...
```

### 8.2 V5B2 — 多角度拍摄管线

```python
# data/multi_view_capture.py
class MultiViewCapture:
    """多机位同步拍摄 + 标定"""
    def capture_pre_firing(self, ceramic_id: str) -> dict:
        """返回 3 视角 RGB + 标定参数"""
        ...

    def capture_post_firing(self, ceramic_id: str) -> dict:
        ...
```

### 8.3 V5B3 — crack_annotator 标注工具 ✅ 已填实（2026-08-25）

```python
# tools/crack_annotator.py（V5B3 已实现 ~680 行）
# 注：v5-α 当前环境未装 streamlit，用 matplotlib 替代（覆盖标注需求）

from tools.crack_annotator import (
    auto_extract_crack,        # 自动骨架提取（Canny + 形态学 + skeletonize）
    save_annotation,            # JSON 序列化（含 image_size/bbox/keypoints/method）
    load_annotation,            # JSON 反序列化
    CrackAnnotatorGUI,          # matplotlib 鼠标交互（左键添加/右键删除/'a'自动/'s'保存）
    batch_auto_extract,         # CLI 批处理（无需 GUI）
)

# 用法 1：自动骨架提取（v5-α 默认）
result = auto_extract_crack(image, min_keypoints=8, max_keypoints=16, verbose=True)
# 返回：{"keypoints": (K, 2), "bbox": (4,), "skeleton": (H, W) bool}

# 用法 2：JSON 持久化
path = save_annotation(
    image_path="img.png",
    keypoints=result["keypoints"],
    bbox=result["bbox"],
    ceramic_id="cart03_layer2",
    skeleton_method="canny_skeletonize",  # 或 "manual"
)
ann = load_annotation(path)
# ann["crack_keypoints"] → np.ndarray (K, 2)

# 用法 3：交互式 GUI
gui = CrackAnnotatorGUI(
    image_path="img.png", ceramic_id="cart03_layer2",
    min_keypoints=8, max_keypoints=16,
)
gui.run()

# CLI 三模式
# $ python tools/crack_annotator.py gui --image img.png --ceramic-id cart03
# $ python tools/crack_annotator.py batch --input-dir data/raw/ --output-dir data/ann/
# $ python tools/crack_annotator.py validate path/to/annotation.json
```

### 8.4 V5B4 — 位置图像 L0/L1 完整管线 ✅ 已填实（2026-08-25）

```python
# data/position_extractor.py（V5B4 已实现 ~410 行）
from data.position_extractor import (
    PositionExtractor, PositionInfo,
    extract_L0, extract_L1, extract_L2, extract_L3, extract_L4,
)

# 5 级降级接口
p_info = PositionExtractor(
    mode="auto",                       # auto / L0 / L1 / L2 / L3 / L4
    images=multi_view_images,          # L0/L1 用（list of np.ndarray H×W×3 uint8）
    checkerboard_inner_corners=(7, 9), # L0 标定板
    square_size_mm=25.0,               # L0 单格尺寸
    kiln_corners=[(x1,y1), ...],       # L0/L1 窑体框
    kiln_visible=True,                 # L1
    ceramic_count=3,                   # L2
    cart_id="cart03", layer_idx=2,     # L3
    n_layers=5,                        # L3
    kiln_layout=None,                  # L3（None → DEFAULT_KILN_LAYOUT）
).extract()

# p_info.level → "L0"/"L1"/"L2"/"L3"/"L4"
# p_info.position_6d → torch.Tensor (6,) [tx, ty, tz, rx, ry, rz]
# p_info.validity_mask → torch.Tensor (6,) True = 该维度有真实数据
# p_info.source_meta → dict 输入数据来源
# p_info.to(device) → 迁移到 GPU

# 单独调用（更灵活）
pos_l3 = extract_L3(cart_id=3, layer_idx=2)  # 完整实现
pos_l4 = extract_L4()                          # 全 0
# extract_L0() / extract_L1() → cv2 缺失时降级到 L3（v5-α stub）
```

### 8.5 V5B5 — 3D FEM 仿真（可选）

```python
# data/fem_3d_simulator.py
class FEM3DSimulator:
    """
    3D 有限元仿真（替代 2D patch 仿真）
    - 求解器：FEniCS / sfepy
    """
    def __init__(
        self,
        mesh_path: str,
        material: dict,
        thermal_profile: dict,
        boundary_conditions: dict,
        solver: str = "fenics",
    ):
        ...

    def solve(
        self,
        time_step_s: float = 60.0,
        total_time_s: float = 42000.0,
        output_interval_s: float = 600.0,
    ) -> dict:
        """
        Returns:
            temperature_3d: (T, N_vertices)
            stress_3d:      (T, N_vertices, 6)
            crack_3d:       (T, N_elements) bool
        """
        ...
```

### 8.6 V5B6 — 磁盘缓存 ✅ 已填实（2026-08-25，npz 实现）

```python
# data/patch_dataset_v5.py（V5B6 已集成，npz 而非 h5py）
from data.patch_dataset_v5 import PatchDatasetV5, collate_v5

# v5-α 用法不变
ds = PatchDatasetV5(patch_size=256, n_samples=100, ...)

# v5-β 加 cache_dir + use_position_image
ds = PatchDatasetV5(
    patch_size=256,
    n_samples=100,
    data_source="simulation",          # v5-α 默认；v5-β 改 "real"
    use_position_image=True,           # 注入 true_position_6d 字段
    cache_dir="logs/v5b6_cache",       # npz 缓存目录
)
# 缓存文件名：sim_cache_p{patch_size}_n{n_samples}_s{seed}_t{thermal_hash}_th{threshold}.npz
# 缓存内容：x_1d / x_2d / true_bbox / true_keypoints / keypoint_mask /
#           true_crack_pixels / pixel_mask / true_position_6d

# collate_v5 仅当所有样本都含 position_6d 时输出
batch = collate_v5([ds[i] for i in range(B)])
batch.get("true_position_6d")  # → (B, 6) 或 None

# 注：h5py 未装，用 npz 等效（单文件 vs 切片访问略慢）
```

### 8.7 V5B7 — team_train_v5.py ✅ 已填实（2026-08-25）

详见 [团队协作训练指南.md §四](团队协作训练指南.md)。

```python
# team_train_v5.py（V5B7 已实现 ~720 行，从 v4 team_train.py 移植 + v5-α 适配）
# CLI 入口：
#   python team_train_v5.py --list-tasks  # 列出 V5A*/V5B* 任务
#   python team_train_v5.py              # 交互式菜单
#   python team_train_v5.py --auto       # 自动执行
#   python team_train_v5.py --import     # 导入队友检查点

import team_train_v5

# 关键函数（继承自 v4）
all_tasks = team_train_v5.load_tasks_from_files()  # 仅 V5* 前缀
hardware_level, total_mem = team_train_v5.get_hardware_level()
completed = team_train_v5.get_completed_tasks()

# 内置任务路由
module_map = team_train_v5._V5_MODULE_TEST_MAP
# {"simulator": ("tests/test_patch_simulator_v5.py",),
#  "heads": ("tests/test_spline_head.py", "tests/test_gp_module.py"),
#  "loss": ("tests/test_ordered_kp_loss.py",),
#  "backbones": ("tests/test_pe_tsnet_v5.py",),
#  "trainer": ("tests/test_trainer_v5_smoke.py",),
#  "annotator": ("tools/crack_annotator.py",),
#  "position": ("data/position_extractor.py",),
#  "cache": ("data/patch_dataset_v5.py",),
#  "team": (None,)}
```

### 8.8 V5B8 — Streamlit v5 推理 GUI

```python
# tools/streamlit_app_v5.py
"""
- 上传烧后图 → 自动 UV 展开 → 推理 → 显示 3D 标注
- 多视图融合
- 位置 6D 显示
"""
```

### 8.9 v5-β 模型扩展

```python
# models/pe_tsnet_multimodal_v5_beta.py（V5B4 新增）
class PETSNetMultimodalV5Beta(nn.Module):
    """v5-β 模型：在 v5-α 基础上加位置图像分支 + UV 映射头"""
    def __init__(
        self,
        image_channels: int = 3,
        use_gp: bool = True,
        use_position_image: bool = True,   # v5-β 新增
        use_uv_inverse: bool = True,       # v5-β 新增
        ...
    ):
        # 复用 v5-α 组件
        self.backbone_2d = ...
        self.fusion = ...
        self.gp = GPModule(...)
        self.head = SplineHead(...)
        # v5-β 新增
        self.position_branch = PositionImageBranch(...)
        self.uv_inverse_head = UVInverseHead(...)
```

### 8.10 v5-β vs v5-α API 兼容性

| v5-α API | v5-β 状态 | 备注 |
|----------|-----------|------|
| `create_v5_model(variant)` | ✅ 兼容 | v5-β 可加 `use_position_image` 等 kwargs |
| `PatchSimulator.simulate()` | ✅ 保留 | v5-β 可选换 `FEM3DSimulator`（V5B5）|
| `PatchDatasetV5` | ✅ 兼容 + 新增 `data_source`/`cache_dir`/`use_position_image` | V5B6 已实现 |
| `collate_v5` | ✅ 兼容 + 新增 `true_position_6d` 字段（条件输出）| 已实现 |
| `TrainerV5` | ✅ 兼容 + 6 个 V5-β 钩子 | 已实现 |
| `OrderedKeypointLoss` | ✅ 兼容 + 新增 `position_6d_loss`/`uv_projection_loss` | 已实现 |
| `SplineHead` | ✅ 兼容 | v5-β 加 UV 逆映射头（V5B1）|
| `GPModule` | ⚠️ 改进 | v5-β 改用 running stats（修复 V5-014/015）|
| `run_train_v5.py` | ✅ 兼容 | v5-β 加 V5-β CLI 参数（use_position_image 等）|
| `team_train_v5.py`（V5B7）| ✅ **新增**（V5B7 已填实）| 从 v4 `team_train.py` 移植 + v5-α 适配 |
| `crack_annotator`（V5B3）| ✅ **新增**（V5B3 已填实）| GUI + CLI batch + 校验 3 模式 |
| `position_extractor`（V5B4）| ✅ **新增**（V5B4 已填实）| 5 级降级 + auto 模式 |

---

## §9. 文档历史

- **2026-08-12（V5A5）**：fork 自 v4.6.11 同名文档，§0 加 v5-α 标注
- **2026-08-24（v5.0.6-alpha）**：完全废弃 v4 内容（`train_model` / `evaluate_model` / `team_train.py` / `freeze_model_backbone` 等），改为 v5-α 完整 API 索引
- **2026-08-24（v5.0.6-alpha）**：新增 §8 v5-β 规划 API（V5B1-V5B8 全部 8 个任务的 API 接口预留）
- **2026-08-25（v5.0.7-alpha，本次）**：
  - §8.3 V5B3 / §8.4 V5B4 / §8.6 V5B6 / §8.7 V5B7 标记已填实，给出实际接口
  - §8.10 兼容性表更新（4 个已填实条目）