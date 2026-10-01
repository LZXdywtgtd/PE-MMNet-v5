"""
v5-α patch 数据集（V5A5 + V5-β 钩子 + V5B6 npz 缓存）

将 PatchSimulator 包装为 PyTorch Dataset。

设计：
- 每个 __getitem__ 调用 PatchSimulator.simulate() 生成一个样本
- 样本包含：
  - x_1d: 温度时序（取 crack 时刻附近的 1D 温度均值）
  - x_2d: 3 通道图像（温度场 + 应力场 + 热力图）
  - true_bbox: (4,) 真值 bbox（[y1, x1, y2, x2] 归一化到 [0,1]）
  - true_keypoints: (K, 2) 真值关键点（K ∈ [min, max]）
  - true_crack_pixels: (N, 2) 真值裂纹像素（来自 crack_mask）

[v5-β 钩子]：
- data_source: "simulation"（v5-α 默认）/ "real"（v5-β 真实数据）
- use_position_image: bool — 注入 position_6d (B,6) 字段
- cache_dir: str | None — 仿真结果 npz 磁盘缓存（V5B6）
  * 缓存键：{patch_size, n_samples, seed, thermal_hash, threshold,
    数据变体标记 dp/sd+gs} → cache_dir/sim_cache_{payload}.npz
  * 变体标记（V5-033 P0）：键不含数据语义 = 静默错位（augfix1/
    V5-033 两次事故根因），故 dp（D'）/sd（stock）与剥框版本 gs
    必须进键；dprime 键 miss 时报错，拒绝现场仿真 stock 冒充

注：
- v5-α 不需要真实 3D 数据；仿真数据已足够训练算法骨架
- v5-β 启动后接入真实数据时，只需替换本 Dataset 即可，trainer 接口不变

详见：
- docs/v5_范围说明.md
- tasks/team_v5_alpha.json 中 V5A5 + V5B6
"""

from __future__ import annotations

import os
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

from data.patch_simulator_v5 import PatchSimulator


def _normalize_field(field: np.ndarray) -> np.ndarray:
    """归一化字段到 [0, 1]"""
    f_min, f_max = field.min(), field.max()
    if f_max - f_min < 1e-9:
        return np.zeros_like(field, dtype=np.float32)
    return ((field - f_min) / (f_max - f_min)).astype(np.float32)


def _thermal_profile_hash(thermal_profile: dict) -> str:
    """热曲线配置 → 短哈希（缓存键用）"""
    keys = sorted(thermal_profile.keys())
    payload = "|".join(f"{k}={thermal_profile[k]}" for k in keys)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


# 数据变体标记（V5-033 P0）：键缺数据语义 → 静默错位（augfix1 用
# stock 数据训了 150ep；V5-033 剥框前旧 GT 被训练链加载）。标记必须
# 进键，且外部生成器数据（dprime）键 miss 时拒绝现场仿真兜底——
# 数据集现算只会产出 stock，用 stock 冒充 dprime 正是事故模式本身。
# gs = gt_frame_strip（V5-032 剥框版 GT）；后续 GT 修复追加新标记。
_DP_TAG = "dp"       # dprime 边界调制 GT（_dprime_generator 产出）
_STOCK_TAG = "sd"    # stock 现场仿真（数据集现算唯一能产出的变体）


def _cache_filename(
    patch_size: int, n_samples: int, seed: int,
    thermal_hash: str, threshold: float,
    data_variant: str = _STOCK_TAG, gt_frame_strip: bool = True,
) -> str:
    """缓存文件名（包含所有影响仿真结果的参数 + 数据变体标记）

    V5-033 P0：th 值曾用明文（60.0），现并入变体段哈希；
    data_variant 区分 dprime/stock；gt_frame_strip 标记剥框版本
    （V5-032 修复后恒 True，False=剥框前旧 GT 口径）。
    """
    frame = "gs" if gt_frame_strip else "og"
    payload = (
        f"p{patch_size}_n{n_samples}_s{seed}_t{thermal_hash}"
        f"_{data_variant}{frame}_h{threshold}"
    )
    return f"sim_cache_{payload}.npz"


class PatchDatasetV5(Dataset):
    """
    v5-α patch 数据集（V5-β 钩子已集成 + V5B6 npz 磁盘缓存）

    每个样本由 PatchSimulator.simulate() 生成。
    默认在 __init__ 时**预计算全部样本**到内存（v5-α MVP 简化）。
    V5B6 启用 cache_dir 时，先尝试加载 npz 缓存，命中则跳过仿真。

    Args:
        patch_size:  patch 像素尺寸（默认 256）
        n_samples:   数据集样本数（默认 100）
        min_kpts:    关键点下界
        max_kpts:    关键点上界
        seq_len:     1D 序列长度（取 crack 时刻附近窗口均值）
        crack_stress_threshold_MPa: 裂纹阈值
        seed:        基础随机种子（每个样本用 seed + idx）
        thermal_profile: 三段曲线配置
        material:    材料参数
        return_pixels: 是否返回 crack 像素（用于覆盖距离 loss）
        precompute:   是否在 __init__ 时预计算（默认 True）
        verbose:      是否打印预计算进度
        ---- v5-β 钩子 ----
        data_source: "simulation"（v5-α）/ "real"（v5-β 占位）
        use_position_image: bool — 注入 true_position_6d 字段（v5-β 占位）
        cache_dir: str | None — 仿真 npz 磁盘缓存目录（v5-β V5B6）
        ---- V5-033 数据变体 ----
        data_variant: 缓存键变体标记。"stock"（默认，现场仿真可复现）/
          "dprime"（外部生成器 GT，只允许命中缓存）
        gt_frame_strip: GT 是否为 V5-032 剥框口径（默认 True；
          False 仅用于复现旧口径，会写入 og 标记键）
    """

    def __init__(
        self,
        patch_size: int = 256,
        n_samples: int = 100,
        min_kpts: int = 8,
        max_kpts: int = 16,
        seq_len: int = 300,
        crack_stress_threshold_MPa: float = 50.0,
        seed: int = 42,
        thermal_profile: dict | None = None,
        material: dict | None = None,
        return_pixels: bool = True,
        precompute: bool = True,
        verbose: bool = False,
        # ---- v5-β 钩子 ----
        data_source: str = "simulation",
        use_position_image: bool = False,
        cache_dir: str | None = None,
        # ---- V5-033 数据变体 ----
        data_variant: str = "stock",
        gt_frame_strip: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.min_kpts = min_kpts
        self.max_kpts = max_kpts
        self.seq_len = seq_len
        self.crack_threshold_MPa = crack_stress_threshold_MPa
        self.seed = seed
        self.thermal_profile = thermal_profile or {}
        self.material = material or {}
        self.return_pixels = return_pixels
        self.verbose = verbose
        # v5-β
        self.data_source = data_source
        self.use_position_image = use_position_image
        self.cache_dir = cache_dir
        # V5-033
        if data_variant not in ("stock", "dprime"):
            raise ValueError(
                f"data_variant 应为 'stock' / 'dprime'，实际 {data_variant!r}"
            )
        if data_variant == "dprime" and cache_dir is None:
            raise ValueError(
                "[V5-033] data_variant='dprime' 必须配 cache_dir："
                "dprime GT 只能来自磁盘缓存，无缓存目录时本数据集"
                "只会现场仿真 stock 数据，冒充 dprime 即 augfix1 事故模式"
            )
        self.data_variant = data_variant
        self.gt_frame_strip = gt_frame_strip

        # 物量单位（米）
        self.physical_size_cm = 5.0
        self.physical_size_m = self.physical_size_cm / 100.0

        # 缓存路径
        self._cache_file: str | None = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            thermal_hash = _thermal_profile_hash(self.thermal_profile)
            fname = _cache_filename(
                patch_size, n_samples, seed, thermal_hash,
                crack_stress_threshold_MPa,
                data_variant=_DP_TAG if self.data_variant == "dprime"
                else _STOCK_TAG,
                gt_frame_strip=self.gt_frame_strip,
            )
            self._cache_file = os.path.join(cache_dir, fname)
            # dprime 数据集现算产不出（现算只有 stock）→ miss 即拒绝，
            # 杜绝 stock 冒充 dprime（augfix1 / V5-033 事故模式）
            if (
                self.data_variant == "dprime"
                and not os.path.exists(self._cache_file)
            ):
                raise FileNotFoundError(
                    f"[V5-033] dprime 缓存未命中且不可现场仿真："
                    f"{self._cache_file} 不存在。dprime GT 只能由 "
                    f"generate_cache_v5.py --dprime 产出后 assemble；"
                    f"请先核对缓存文件名/参数（seed/th/剥框标记），"
                    f"缺失时用生成器补齐——绝不静默用 stock 数据顶替。"
                )

        # 预计算缓存
        self._cache: list[dict] | None = None
        if precompute:
            self._precompute()

    def _precompute(self) -> None:
        """预计算所有样本（优先用 npz 缓存）"""
        # V5B6: 尝试加载 npz 磁盘缓存
        if self._cache_file is not None and os.path.exists(self._cache_file):
            if self.verbose:
                print(f"  [PatchDatasetV5] 加载 npz 缓存: {self._cache_file}")
            self._cache = self._load_from_npz(self._cache_file)
            return

        # stock 现场仿真（无缓存时唯一合法路径）；dprime 已在 __init__ 拒绝
        self._cache = []
        for idx in range(self.n_samples):
            if self.verbose:
                print(f"  [PatchDatasetV5] 预计算样本 {idx + 1}/{self.n_samples}")
            self._cache.append(self._generate_sample(idx))

        # V5B6: 仿真完成 → 写入 npz 缓存
        if self._cache_file is not None:
            self._save_to_npz(self._cache_file, self._cache)
            if self.verbose:
                print(f"  [PatchDatasetV5] 已写入 npz 缓存: {self._cache_file}")

    def _save_to_npz(self, path: str, cache: list[dict]) -> None:
        """把内存缓存序列化到 npz 文件"""
        n = len(cache)
        x_1d = np.stack([c["x_1d"].numpy() for c in cache], axis=0)
        x_2d = np.stack([c["x_2d"].numpy() for c in cache], axis=0)
        true_bbox = np.stack([c["true_bbox"].numpy() for c in cache], axis=0)
        # 变长关键点：用一个 mask 标记
        max_K = max(c["true_keypoints"].size(0) for c in cache)
        if max_K > 0:
            true_kpts = np.full((n, max_K, 2), -1.0, dtype=np.float32)
            kpt_mask = np.zeros((n, max_K), dtype=bool)
            for i, c in enumerate(cache):
                k = c["true_keypoints"].size(0)
                if k > 0:
                    true_kpts[i, :k] = c["true_keypoints"].numpy()
                    kpt_mask[i, :k] = True
        else:
            true_kpts = np.zeros((n, 0, 2), dtype=np.float32)
            kpt_mask = np.zeros((n, 0), dtype=bool)
        max_N = max(c["true_crack_pixels"].size(0) for c in cache)
        if max_N > 0:
            true_px = np.full((n, max_N, 2), -1.0, dtype=np.float32)
            px_mask = np.zeros((n, max_N), dtype=bool)
            for i, c in enumerate(cache):
                m = c["true_crack_pixels"].size(0)
                if m > 0:
                    true_px[i, :m] = c["true_crack_pixels"].numpy()
                    px_mask[i, :m] = True
        else:
            true_px = np.zeros((n, 0, 2), dtype=np.float32)
            px_mask = np.zeros((n, 0), dtype=bool)
        # v5-β: position_6d（None → 全 0 + 全 False mask）
        if self.use_position_image and all(
            c.get("true_position_6d") is not None for c in cache
        ):
            true_pos6d = np.stack(
                [c["true_position_6d"].numpy() for c in cache], axis=0,
            ).astype(np.float32)
        else:
            true_pos6d = np.zeros((n, 6), dtype=np.float32)

        np.savez_compressed(
            path,
            x_1d=x_1d, x_2d=x_2d, true_bbox=true_bbox,
            true_keypoints=true_kpts, keypoint_mask=kpt_mask,
            true_crack_pixels=true_px, pixel_mask=px_mask,
            true_position_6d=true_pos6d,
        )

    def _load_from_npz(self, path: str) -> list[dict]:
        """从 npz 加载 → 内存缓存

        2026-09-29 修复：原先直接返回 padding 后的定长数组而未读 mask，
        -1 填充行被当作真实关键点/像素进入 collate，与现算路径（变长）
        语义不一致。现按 keypoint_mask/pixel_mask 裁剪回变长形状。
        """
        data = np.load(path)
        # npz 是 lazy 解压：循环内 data["x_2d"][i] 每次都整组重新解压
        # （1000 样本 = 200 GB 累计流量 → 分配器碎片化 OOM，2026-09-29）。
        # 循环外一次性物化全部成员（~330 MB @128px/1000），循环内只切片。
        members = {k: data[k] for k in data.files}
        data.close()
        kpt_mask = members["keypoint_mask"]
        px_mask = members["pixel_mask"]
        cache = []
        for i in range(members["x_1d"].shape[0]):
            k = int(kpt_mask[i].sum())
            m = int(px_mask[i].sum())
            sample = {
                "x_1d": torch.from_numpy(members["x_1d"][i]),
                "x_2d": torch.from_numpy(members["x_2d"][i]),
                "true_bbox": torch.from_numpy(members["true_bbox"][i]),
                "true_keypoints": torch.from_numpy(
                    members["true_keypoints"][i][:k].copy()
                ),
                "true_crack_pixels": torch.from_numpy(
                    members["true_crack_pixels"][i][:m].copy()
                ),
            }
            # v5-β position_6d
            if (
                self.use_position_image
                and "true_position_6d" in members
            ):
                sample["true_position_6d"] = torch.from_numpy(
                    members["true_position_6d"][i]
                )
            cache.append(sample)
        return cache

    def __len__(self) -> int:
        return self.n_samples

    def _generate_sample(self, idx: int) -> dict:
        """生成单个样本（不依赖缓存）"""
        sim_seed = self.seed + idx
        sim = PatchSimulator(
            patch_size=self.patch_size,
            physical_size_cm=self.physical_size_cm,
            thermal_profile=self.thermal_profile,
            material=self.material,
            crack_stress_threshold_MPa=self.crack_threshold_MPa,
            min_keypoints=self.min_kpts,
            max_keypoints=self.max_kpts,
            seed=sim_seed,
        )
        result = sim.simulate(verbose=False)

        H = W = self.patch_size

        # 1D 时序：取 T_history 的最后 seq_len 个点的全局平均
        T_hist = result["temperature_field"]  # (T_samples, H, W)
        if len(T_hist) >= self.seq_len:
            T_seq = T_hist[-self.seq_len:].mean(axis=(1, 2))
        else:
            T_seq = T_hist.mean(axis=(1, 2))
            T_seq = np.pad(
                T_seq,
                (self.seq_len - len(T_seq), 0),
                mode="edge",
            )

        T_seq = _normalize_field(T_seq)

        # 2D 图像：3 通道
        T_final = T_hist[-1] if len(T_hist) > 0 else result["heatmap"]
        T_field_norm = _normalize_field(T_final)

        S_hist = result["stress_field"]
        S_final = S_hist[-1] if len(S_hist) > 0 else np.zeros((H, W))
        S_field_norm = _normalize_field(S_final)

        H_map = result["heatmap"].astype(np.float32)

        x_2d = np.stack([T_field_norm, S_field_norm, H_map], axis=0)

        # bbox：仿真器 _mask_to_bbox 已输出归一化 xyxy [x1,y1,x2,y2]，直接用
        # （2026-09-28 修复双重归一化：原代码再除 H/W 且按 yxyx 换序，导致 bbox 缩水 ~H 倍）
        bbox_xyxy = result["crack_bbox"].astype(np.float32).copy()

        # 关键点：(y, x) 归一化
        kpts = result["crack_keypoints"].astype(np.float32).copy()
        kpts[:, 0] /= H
        kpts[:, 1] /= W

        # 裂纹像素
        crack_mask = result["crack_mask"]
        if self.return_pixels:
            ys, xs = np.where(crack_mask > 0)
            if len(ys) > 0:
                true_pixels = np.stack([
                    ys.astype(np.float32) / H,
                    xs.astype(np.float32) / W,
                ], axis=1)
            else:
                true_pixels = np.zeros((0, 2), dtype=np.float32)
        else:
            true_pixels = np.zeros((0, 2), dtype=np.float32)

        sample = {
            "x_1d": torch.from_numpy(T_seq),
            "x_2d": torch.from_numpy(x_2d),
            "true_bbox": torch.from_numpy(bbox_xyxy),
            "true_keypoints": torch.from_numpy(kpts),
            "true_crack_pixels": torch.from_numpy(true_pixels),
            "metadata": result["metadata"],
        }

        # v5-β: 位置 6D（仿真数据无 → 全 0）
        if self.use_position_image:
            sample["true_position_6d"] = torch.zeros(6, dtype=torch.float32)

        return sample

    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            return self._cache[idx]
        return self._generate_sample(idx)


def collate_v5(batch: list[dict]) -> dict:
    """
    v5-α 自定义 collate 函数（v5-β 钩子已集成）

    处理变长数据（true_keypoints, true_crack_pixels 不同长度）：
    - 用 padding（-1 标记填充）保留 batch 维度
    - 用 mask 标记有效位置

    v5-β 钩子：
    - 若 batch 含 'true_position_6d'，则堆叠为 (B, 6)
    - 若 use_position_image=False 或缺失字段，则输出中不包含此键

    Returns:
        dict:
            x_1d:        (B, seq_len)
            x_2d:        (B, 3, H, W)
            true_bbox:   (B, 4)
            true_keypoints: (B, max_K, 2) — padding 到最大 K
            keypoint_mask: (B, max_K) — True = 有效
            true_crack_pixels: (B, max_N, 2) — padding 到最大 N
            pixel_mask:    (B, max_N) — True = 有效
            true_position_6d: (B, 6) — 仅当所有样本都有此字段时存在
    """
    B = len(batch)

    x_1d = torch.stack([b["x_1d"] for b in batch], dim=0)
    x_2d = torch.stack([b["x_2d"] for b in batch], dim=0)
    true_bbox = torch.stack([b["true_bbox"] for b in batch], dim=0)

    # 变长关键点 — padding
    Ks = [b["true_keypoints"].size(0) for b in batch]
    max_K = max(Ks)
    if max_K > 0:
        true_kpts = torch.full((B, max_K, 2), -1.0, dtype=torch.float32)
        kpt_mask = torch.zeros(B, max_K, dtype=torch.bool)
        for i, b in enumerate(batch):
            k = b["true_keypoints"].size(0)
            if k > 0:
                true_kpts[i, :k] = b["true_keypoints"]
                kpt_mask[i, :k] = True
    else:
        true_kpts = torch.zeros(B, 0, 2)
        kpt_mask = torch.zeros(B, 0, dtype=torch.bool)

    # 变长裂纹像素 — padding
    Ns = [b["true_crack_pixels"].size(0) for b in batch]
    max_N = max(Ns) if Ns else 0
    if max_N > 0:
        true_pixels = torch.full((B, max_N, 2), -1.0, dtype=torch.float32)
        pixel_mask = torch.zeros(B, max_N, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b["true_crack_pixels"].size(0)
            if n > 0:
                true_pixels[i, :n] = b["true_crack_pixels"]
                pixel_mask[i, :n] = True
    else:
        true_pixels = torch.zeros(B, 0, 2)
        pixel_mask = torch.zeros(B, 0, dtype=torch.bool)

    out = {
        "x_1d": x_1d,
        "x_2d": x_2d,
        "true_bbox": true_bbox,
        "true_keypoints": true_kpts,
        "keypoint_mask": kpt_mask,
        "true_crack_pixels": true_pixels,
        "pixel_mask": pixel_mask,
    }

    # v5-β: position_6d（仅当所有样本都有此字段时输出）
    if all("true_position_6d" in b for b in batch):
        out["true_position_6d"] = torch.stack(
            [b["true_position_6d"] for b in batch], dim=0,
        )

    return out


# ============================================================
#  冒烟测试
# ============================================================

def _smoke_test():
    print("=" * 60)
    print("PatchDatasetV5 冒烟测试（小尺寸 + 短周期加速）")
    print("=" * 60)

    # 用小 patch + 短热曲线，让 1 次仿真 < 5s
    # 默认：11.7h 周期 + 256x256 → 几十分钟
    # 测试用：1 min 周期 + 64x64 patch → < 1s
    fast_profile = {
        "ramp_up_c_per_min": 1260.0,    # 1 min ramp
        "soak_temp_c": 1280.0,
        "soak_duration_min": 1.0,
        "cool_down_c_per_min": 1260.0,  # 1 min cool
    }

    # v5-β 钩子测试：cache_dir + use_position_image
    cache_dir = "logs/_test_cache"
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)

    # 1) 默认（无缓存 + 无 position_6d）
    ds = PatchDatasetV5(
        patch_size=64,
        n_samples=2,
        seed=42,
        thermal_profile=fast_profile,
        verbose=True,
    )
    print(f"\n[OK] 数据集大小: {len(ds)}")

    sample = ds[0]
    print(f"  x_1d:        {tuple(sample['x_1d'].shape)}")
    print(f"  x_2d:        {tuple(sample['x_2d'].shape)}")
    print(f"  true_bbox:   {tuple(sample['true_bbox'].shape)}, values={sample['true_bbox'].tolist()}")
    print(f"  true_keypoints: {tuple(sample['true_keypoints'].shape)}")
    print(f"  true_crack_pixels: {tuple(sample['true_crack_pixels'].shape)}")
    assert "true_position_6d" not in sample, "v5-α 默认不应有 position_6d"
    print(f"  [v5-β] true_position_6d: 不存在（符合预期）")
    print(f"  metadata:    peak={sample['metadata']['peak_stress_MPa']:.2f} MPa, "
          f"K={sample['metadata']['n_keypoints']}")

    # collate 测试
    batch = collate_v5([ds[0], ds[1]])
    print(f"\n[Collate]")
    print(f"  x_1d:        {tuple(batch['x_1d'].shape)}")
    print(f"  x_2d:        {tuple(batch['x_2d'].shape)}")
    print(f"  true_keypoints: {tuple(batch['true_keypoints'].shape)}, mask sum={batch['keypoint_mask'].sum().item()}")
    print(f"  true_crack_pixels: {tuple(batch['true_crack_pixels'].shape)}, mask sum={batch['pixel_mask'].sum().item()}")
    assert "true_position_6d" not in batch, "默认 collate 不应包含 position_6d"

    # 2) npz 缓存测试：第 1 次仿真写入缓存 → 第 2 次直接加载
    print(f"\n[npz 缓存测试]")
    ds1 = PatchDatasetV5(
        patch_size=64, n_samples=2, seed=42,
        thermal_profile=fast_profile, cache_dir=cache_dir, verbose=True,
    )
    cache_files = os.listdir(cache_dir)
    print(f"  第 1 次完成，缓存文件: {cache_files}")
    assert len(cache_files) == 1, f"应写 1 个 npz，实际 {len(cache_files)}"

    ds2 = PatchDatasetV5(
        patch_size=64, n_samples=2, seed=42,
        thermal_profile=fast_profile, cache_dir=cache_dir, verbose=True,
    )
    sample2 = ds2[0]
    assert torch.allclose(sample["x_1d"], sample2["x_1d"]), "缓存命中应得到相同数据"
    print(f"  [OK] 缓存命中，x_1d 一致")

    # 3) use_position_image 测试
    print(f"\n[use_position_image 测试]")
    ds3 = PatchDatasetV5(
        patch_size=64, n_samples=2, seed=42,
        thermal_profile=fast_profile, use_position_image=True, verbose=False,
    )
    sample3 = ds3[0]
    assert "true_position_6d" in sample3, "use_position_image=True 应注入字段"
    assert sample3["true_position_6d"].shape == (6,), f"应 (6,)，实际 {tuple(sample3['true_position_6d'].shape)}"
    print(f"  [OK] true_position_6d 注入: shape={tuple(sample3['true_position_6d'].shape)}")
    batch3 = collate_v5([ds3[0], ds3[1]])
    assert "true_position_6d" in batch3
    assert batch3["true_position_6d"].shape == (2, 6)
    print(f"  [OK] collate_v5 含 position_6d: {tuple(batch3['true_position_6d'].shape)}")

    # 清理
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)

    print("\n[ALL PASS] ✅")


if __name__ == "__main__":
    _smoke_test()