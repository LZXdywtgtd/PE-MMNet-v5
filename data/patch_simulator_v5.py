"""
patch 仿真器（V5A1 — S1 + S4 + S7 集成）

V5A1 整合模块：在 256×256 patch 上跑完整热-应力-裂纹仿真。

工作流程：
1. 生成三段升降温曲线（S4：thermal_profile.py）
2. 每步用 FDM 更新温度场（S1：多方向热交换）
3. 边界条件叠加对流 + 辐射热流（S2：surface_radiation.py）
4. 由温差累积 Maxwell 应力（S3：viscoelastic.py）
5. 应力峰值时刻提取 crack mask + 关键点（S7）

详见：
- docs/v5_仿真升级清单.md S1–S4 + S7
- tasks/team_v5_alpha.json 中 V5A1
"""

from __future__ import annotations

from collections import deque

import numpy as np

from data.thermal_profile import generate_thermal_profile
from data.surface_radiation import SurfaceRadiation
from data.viscoelastic import MaxwellStress


# 陶瓷材料默认参数（v5-α MVP，参考日用陶瓷典型值）
DEFAULT_MATERIAL = {
    "thermal_conductivity": 1.5,    # λ (W/(m·K))，陶瓷 ~1–3
    "specific_heat": 800.0,         # c (J/(kg·K))，陶瓷 ~700–900
    "density": 2400.0,              # ρ (kg/m³)，陶瓷 ~2000–2500
    "convection_coeff": 15.0,       # h (W/(m²·K))，自然对流 ~10–25
}


class PatchSimulator:
    """
    256×256 patch 仿真器

    简化假设：
    - 2D 平面（patch = 陶瓷表面抽象）
    - 显式 Euler 时间步进
    - CFL 条件检查
    """

    def __init__(
        self,
        patch_size: int = 256,
        physical_size_cm: float = 5.0,  # patch 物理尺寸（cm）
        thermal_profile: dict | None = None,
        boundary_config: dict | None = None,
        material: dict | None = None,
        dt_s: float | None = None,      # 时间步长（秒），None=自适应
        crack_stress_threshold_MPa: float = 50.0,  # 裂纹萌生阈值
        min_keypoints: int = 8,
        max_keypoints: int = 16,
        seed: int = 42,
        max_history_samples: int = 600,  # 历史采样截尾（None=不截尾）
    ):
        """
        Args:
            patch_size:         patch 像素尺寸（默认 256）
            physical_size_cm:   patch 物理尺寸（默认 5cm）
            thermal_profile:    三段曲线配置（None=用默认）
            boundary_config:    边界条件开关（None=全开）
            material:           材料参数（None=默认陶瓷）
            dt_s:               时间步长（None=按 CFL 自动）
            crack_stress_threshold_MPa: 裂纹萌生应力阈值
            min_keypoints:      关键点下界
            max_keypoints:      关键点上界
            seed:               随机种子（裂纹形状随机化）
            max_history_samples: 历史采样截尾帧数（None=不截尾，旧行为）
        """
        self.patch_size = patch_size
        self.physical_size_cm = physical_size_cm
        self.physical_size_m = physical_size_cm / 100.0
        self.thermal_profile_cfg = thermal_profile or {}
        self.max_history_samples = max_history_samples
        self.boundary_cfg = boundary_config or {
            "multi_direction_heat": True,
            "surface_radiation": True,
            "stress_relaxation": True,
        }
        self.material = {**DEFAULT_MATERIAL, **(material or {})}
        self.crack_threshold_Pa = crack_stress_threshold_MPa * 1e6
        self.min_keypoints = min_keypoints
        self.max_keypoints = max_keypoints
        self.seed = seed

        # 物理量
        self.lam = self.material["thermal_conductivity"]      # λ
        self.cp = self.material["specific_heat"]              # c
        self.rho = self.material["density"]                   # ρ
        self.alpha_thermal = self.lam / (self.rho * self.cp)  # α = λ/(ρc) 热扩散率
        self.h_conv = self.material["convection_coeff"]       # h

        # 空间步长（每像素）
        self.dx = self.physical_size_m / patch_size

        # CFL 自适应时间步长（2D 显式）
        # 稳定条件：dt ≤ dx² / (4α)
        cfl_dt = self.dx ** 2 / (4 * self.alpha_thermal)
        self.dt = dt_s if dt_s is not None else cfl_dt * 0.8  # 80% CFL 上限
        if dt_s is not None and dt_s > cfl_dt:
            print(f"[WARN] dt={dt_s:.2e}s 超过 CFL={cfl_dt:.2e}s，可能不稳定")

        # 边界条件工具
        self.radiation = SurfaceRadiation(emissivity=0.9)

        # 应力松弛工具
        self.stress_solver = MaxwellStress()

        # 状态变量
        self.T_field: np.ndarray | None = None       # (H, W) 当前温度
        self.sigma_field: np.ndarray | None = None   # (H, W) 当前应力
        self.t_history: list[float] = []             # 时间序列
        self._T_hist_buf: deque | None = None        # 温度历史（有界缓冲）
        self._sigma_hist_buf: deque | None = None    # 应力历史（有界缓冲）
        self._sigma_max_field: np.ndarray | None = None  # (H, W) 全程峰值拉应力场

        # ── 样本多样性随机化（2026-09-28：修复确定性仿真退化）──
        # 原 _rng 只用于关键点数，同一 (patch, 曲线, dt) 下所有样本物理全同。
        # 现在每实例用 rng 抽取批次差异因子（坯体材质/表面状态不均匀）：
        #   E/α_T ±20%、τ ±30%、h ±30%、ε U(0.85,0.95)
        # 注意：必须在 stress_solver 构造后重建它，使随机 E/α_T/τ 生效。
        rng = np.random.default_rng(seed)
        u = rng.uniform
        self.stress_solver = MaxwellStress(
            young_modulus_GPa=200.0 * u(0.8, 1.2),
            thermal_expansion=8e-6 * u(0.8, 1.2),
            relaxation_time_s=60.0 * u(0.7, 1.3),
        )
        self.h_conv *= u(0.7, 1.3)
        self.radiation.emissivity = u(0.85, 0.95)
        # 初始温度噪声幅度 N(0, 2℃)（坯体初始温度不均）
        self._init_temp_noise_std_c = 2.0
        self._init_rng = np.random.default_rng(seed + 1)
        self._rng = rng  # 沿用原用途（关键点数随机化）

    # ============================================================
    #  主仿真入口
    # ============================================================

    def simulate(
        self,
        sample_interval_steps: int = 60,
        verbose: bool = False,
    ) -> dict:
        """
        运行完整烧制过程仿真

        Args:
            sample_interval_steps: 每 N 步采样一次历史
            verbose:                是否打印进度

        Returns:
            dict:
                temperature_field:    (T_samples, H, W)  温度场历史（采样，
                                      最多保留 max_history_samples 帧）
                stress_field:         (T_samples, H, W)  应力场历史（采样）
                heatmap:              (H, W)             最终温度场（输入给模型）
                crack_mask:           (H, W) uint8       像素级裂纹 mask（GT）
                crack_keypoints:      (K, 2)             关键点 GT，K ∈ [8, 16]
                crack_bbox:           (4,)               bbox GT
                metadata:             dict               配置元数据
        """
        # 1. 初始化温度场（室温 + 坯体初始温度不均噪声）
        H = W = self.patch_size
        self.T_field = np.full((H, W), 20.0, dtype=np.float64)  # 20°C
        if self._init_temp_noise_std_c > 0:
            self.T_field += self._init_rng.normal(
                0.0, self._init_temp_noise_std_c, size=(H, W)
            )
        self.sigma_field = np.zeros((H, W), dtype=np.float64)
        self._sigma_max_field = np.zeros((H, W), dtype=np.float64)  # 峰值拉应力场

        # 历史采样缓冲（截尾：只保留最后 max_history_samples 帧）
        # 2026-09-29：默认曲线 128px 全程 6.9 万帧 ≈ 18 GB，多进程 OOM 根因；
        # 模型只用最后 300 帧（x_1d）+ 最终帧（x_2d），截尾不影响模型可见输出。
        if self.max_history_samples is not None:
            self._T_hist_buf = deque(maxlen=self.max_history_samples)
            self._sigma_hist_buf = deque(maxlen=self.max_history_samples)
        else:
            self._T_hist_buf = []
            self._sigma_hist_buf = []
        self.t_history = []

        # 2. 生成三段曲线
        times_s, target_temps_c = generate_thermal_profile(
            **self.thermal_profile_cfg,
            dt_s=self.dt,
        )
        # 将 °C 边界条件存为公共属性，方便热流计算
        self._target_temps_c = target_temps_c
        self._times_s = times_s

        # 3. 时间步进
        n_steps = len(times_s)
        if verbose:
            print(f"[PatchSimulator] 开始仿真：{n_steps} 步，dt={self.dt:.2e}s，"
                  f"总时长={times_s[-1]/3600:.2f}h")

        for step in range(n_steps):
            T_target = target_temps_c[step]
            self._step_heat(self.T_field, T_target)
            # 应力更新（松弛）
            if step > 0:
                dT_field = self.T_field - self._T_prev_for_stress
                self.sigma_field = self.stress_solver.update_step(
                    self.sigma_field, dT_field, self.dt
                )
            self._T_prev_for_stress = self.T_field.copy()

            # 采样历史
            if step % sample_interval_steps == 0 or step == n_steps - 1:
                self.t_history.append(times_s[step])
                self._T_hist_buf.append(self.T_field.copy())
                self._sigma_hist_buf.append(self.sigma_field.copy())

            # 累积峰值拉应力场（每步更新）
            np.maximum(self._sigma_max_field, self.sigma_field,
                       out=self._sigma_max_field)

        if verbose:
            print(f"[PatchSimulator] 仿真完成：峰值拉应力 = "
                  f"{self._sigma_max_field.max()/1e6:.1f} MPa")

        # 4. 生成 GT（裂纹 mask + 关键点 + bbox）
        # 用"全程峰值拉应力场"判定裂纹萌生（裂纹在升温/保温阶段萌生）
        crack_mask = self._extract_crack_mask(self._sigma_max_field)
        keypoints, bbox = self._extract_keypoints(crack_mask)

        # 5. 构造 heatmap（最终温度场，归一化到 [0, 1]）
        heatmap = self._build_heatmap(self.T_field)

        return {
            "temperature_field": np.stack(self._T_hist_buf, axis=0),
            "stress_field":      np.stack(self._sigma_hist_buf, axis=0),
            "heatmap":           heatmap,
            "crack_mask":        crack_mask,
            "crack_keypoints":   keypoints,
            "crack_bbox":        bbox,
            "metadata": {
                "patch_size": self.patch_size,
                "physical_size_cm": self.physical_size_cm,
                "dt_s": self.dt,
                "n_steps": n_steps,
                "total_time_h": times_s[-1] / 3600,
                "peak_stress_MPa": float(self._sigma_max_field.max() / 1e6),
                "n_keypoints": len(keypoints),
            },
        }

    # ============================================================
    #  S1: 多方向热交换（FDM 显式 Euler）
    # ============================================================

    def _step_heat(self, T: np.ndarray, T_target: float) -> None:
        """
        单步温度更新（含 S1 多方向 + S2 辐射边界）

        控制方程：
            ∂T/∂t = α · ∇²T
        边界条件（每边）：
            −λ · ∂T/∂n = h · (T − T_∞) + ε · σ_SB · (T⁴ − T_∞⁴)
        """
        T_new = T.copy()
        alpha = self.alpha_thermal
        dx = self.dx
        dt = self.dt

        # 内部单元：标准 5 点 FDM 拉普拉斯
        T_new[1:-1, 1:-1] = (
            T[1:-1, 1:-1]
            + alpha * dt / dx ** 2
            * (
                T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]
                + T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]
            )
        )

        # 边界单元：Robin 边界条件（对流 + 辐射）
        T_new = self._apply_robin_boundary(T_new, T, T_target)

        # 防止温度下溢（绝对零度）
        np.clip(T_new, 0, None, out=T_new)

        # 写回
        T[:] = T_new

    def _apply_robin_boundary(
        self, T_new: np.ndarray, T_old: np.ndarray, T_target: float
    ) -> np.ndarray:
        """
        应用 Robin 边界（对流 + 辐射）

        边界单元温度更新：
            T_new[边界] = T_old[边界] + dt/(ρ·c·L) · [h·(T_target − T) + q_rad]

        其中 L = dx/2（边界单元到外部环境的特征距离）
        """
        dt = self.dt
        rho_cp = self.rho * self.cp  # 体积热容
        L = self.dx / 2.0  # 边界单元特征距离

        # 环境温度取目标温度（窑内环境模拟）
        T_env_c = T_target
        T_env_K = self.radiation.celsius_to_kelvin(T_env_c)

        # 对每条边应用边界条件
        for edge_slice in [
            (0, slice(None)),         # top
            (-1, slice(None)),        # bottom
            (slice(None), 0),         # left
            (slice(None), -1),        # right
        ]:
            T_edge = T_old[edge_slice]
            T_edge_K = self.radiation.celsius_to_kelvin(T_edge)

            # 对流热流（朝向环境）
            q_conv = self.h_conv * (T_env_c - T_edge)

            # 辐射热流（朝向环境）
            q_rad = self.radiation.heat_flux(T_edge_K, T_env_K)

            # 总热流密度（W/m²）
            q_total = q_conv + q_rad

            # 温度增量
            dT = dt * q_total / (rho_cp * L)

            T_new[edge_slice] = T_edge + dT

        return T_new

    # ============================================================
    #  S7: 像素级裂纹 mask + 关键点提取
    # ============================================================

    def _extract_crack_mask(self, sigma: np.ndarray) -> np.ndarray:
        """
        从应力场生成裂纹 mask

        简化判据（v5-α MVP）：
        - 阈值化：σ > threshold → 裂纹
        - 形态学清理：去小连通域 + 闭运算连接
        - 保留最长连通裂纹

        Args:
            sigma: (H, W) — 应力场（Pa）

        Returns:
            mask: (H, W) uint8 — 0/255 裂纹 mask
        """
        from scipy import ndimage

        # 1. 阈值化
        mask = (sigma > self.crack_threshold_Pa).astype(np.uint8)

        if mask.sum() == 0:
            # 无裂纹：返回空 mask（不影响训练）
            return mask * 255

        # 2. 连通域标记
        labeled, n_features = ndimage.label(mask)

        # 3. 保留最大连通域
        if n_features > 1:
            sizes = ndimage.sum(mask, labeled, range(1, n_features + 1))
            largest_label = int(np.argmax(sizes)) + 1
            mask = (labeled == largest_label).astype(np.uint8)

        # 4. 闭运算（连接断裂处）
        mask = ndimage.binary_closing(mask, iterations=2).astype(np.uint8)

        # 5. 形态学细化（可选）
        # skeleton = morphology.skeletonize(mask > 0)

        return mask * 255

    def _extract_keypoints(
        self, crack_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        从裂纹 mask 提取关键点 + bbox

        流程：
        1. 骨架化
        2. 找最长路径
        3. 等弧长采样 8–16 点
        4. bbox = mask 包络

        Args:
            crack_mask: (H, W) uint8 — 裂纹 mask

        Returns:
            keypoints: (K, 2) — 关键点坐标 (y, x)，K ∈ [min_kpts, max_kpts]
            bbox: (4,) — [x1, y1, x2, y2] 归一化角点
        """
        from scipy import ndimage
        from skimage import morphology

        if crack_mask.sum() == 0:
            # 无裂纹：返回中心点 + 退化 bbox
            cx = cy = self.patch_size // 2
            return (
                np.array([[cy, cx]] * self.min_keypoints, dtype=np.float32),
                np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            )

        # 1. 骨架化
        skeleton = morphology.skeletonize(crack_mask > 0)

        # 2. 找所有骨架点
        ys, xs = np.where(skeleton)
        if len(ys) < 2:
            # 骨架点太少（罕见）
            return (
                np.column_stack([ys, xs]).astype(np.float32),
                self._mask_to_bbox(crack_mask),
            )

        # 3. 找两个端点（骨架中只有 1 个邻居的点）
        endpoints = self._find_endpoints(skeleton)

        if len(endpoints) >= 2:
            # 4. 沿骨架追踪从端点 1 到端点 2 的最长路径
            path = self._trace_path(skeleton, endpoints[0], endpoints[1])
        else:
            # 端点不足：取所有骨架点
            path = list(zip(ys, xs))

        if len(path) < 2:
            return (
                np.column_stack([ys, xs]).astype(np.float32),
                self._mask_to_bbox(crack_mask),
            )

        # 5. 等弧长采样 K 个点
        K = self._rng.integers(self.min_keypoints, self.max_keypoints + 1)
        keypoints = self._resample_arclength(path, K)

        # 6. bbox
        bbox = self._mask_to_bbox(crack_mask)

        return keypoints.astype(np.float32), bbox.astype(np.float32)

    @staticmethod
    def _find_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
        """找骨架端点（邻居数为 1 的点）"""
        from scipy import ndimage

        # 8 邻域核
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = ndimage.convolve(
            skeleton.astype(np.uint8), kernel, mode="constant"
        ) * skeleton

        ys, xs = np.where(neighbor_count == 1)
        return [(int(y), int(x)) for y, x in zip(ys, xs)]

    @staticmethod
    def _trace_path(
        skeleton: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """BFS 追踪从 start 到 end 的骨架路径"""
        from collections import deque

        H, W = skeleton.shape
        visited = np.zeros_like(skeleton, dtype=bool)
        visited[start] = True
        queue = deque([(start, [start])])

        while queue:
            (cy, cx), path = queue.popleft()
            if (cy, cx) == end:
                return path

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if skeleton[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append(((ny, nx), path + [(ny, nx)]))

        # 找不到路径：返回 BFS 访问到的最长路径
        return []

    @staticmethod
    def _resample_arclength(
        path: list[tuple[int, int]], K: int
    ) -> np.ndarray:
        """沿路径等弧长采样 K 个点"""
        if len(path) < 2:
            return np.array(path, dtype=np.float32).reshape(-1, 2)

        pts = np.array(path, dtype=np.float32)  # (N, 2)
        # 累积弧长
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_lens = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum_lens[-1]

        if total_len < 1e-6:
            return np.tile(pts[0], (K, 1))

        # 等弧长目标位置
        target_lens = np.linspace(0, total_len, K)

        # 在路径上插值
        result = np.zeros((K, 2), dtype=np.float32)
        for i, target in enumerate(target_lens):
            # 找到 target 所在段
            idx = np.searchsorted(cum_lens, target) - 1
            idx = max(0, min(idx, len(seg_lens) - 1))
            seg_start_len = cum_lens[idx]
            t = (target - seg_start_len) / max(seg_lens[idx], 1e-9)
            t = np.clip(t, 0, 1)
            result[i] = pts[idx] + t * (pts[idx + 1] - pts[idx])

        return result

    def _mask_to_bbox(self, crack_mask: np.ndarray) -> np.ndarray:
        """mask → 归一化 bbox [x1, y1, x2, y2]"""
        ys, xs = np.where(crack_mask > 0)
        if len(ys) == 0:
            return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        x1, x2 = xs.min() / self.patch_size, xs.max() / self.patch_size
        y1, y2 = ys.min() / self.patch_size, ys.max() / self.patch_size
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _build_heatmap(T_field: np.ndarray) -> np.ndarray:
        """温度场 → 归一化 heatmap（输入给模型）"""
        t_min, t_max = T_field.min(), T_field.max()
        if t_max - t_min < 1e-9:
            return np.zeros_like(T_field, dtype=np.float32)
        return ((T_field - t_min) / (t_max - t_min)).astype(np.float32)


if __name__ == "__main__":
    # 端到端冒烟测试（小尺寸 + 少步数）
    print("=" * 60)
    print("PatchSimulator 冒烟测试")
    print("=" * 60)

    sim = PatchSimulator(
        patch_size=64,                    # 缩小到 64x64 加速
        physical_size_cm=2.0,             # 物理尺寸 2cm
        thermal_profile={
            "ramp_up_c_per_min": 50.0,    # 加速升温（冒烟测试）
            "soak_temp_c": 200.0,         # 降低保温温度（冒烟测试）
            "soak_duration_min": 1.0,
            "cool_down_c_per_min": 30.0,
        },
        crack_stress_threshold_MPa=10.0,  # 降低阈值
    )

    result = sim.simulate(sample_interval_steps=200, verbose=True)

    print(f"\n输出统计：")
    print(f"  temperature_field: {result['temperature_field'].shape}")
    print(f"  stress_field:      {result['stress_field'].shape}")
    print(f"  heatmap:           {result['heatmap'].shape}, "
          f"range [{result['heatmap'].min():.3f}, {result['heatmap'].max():.3f}]")
    print(f"  crack_mask:        {result['crack_mask'].shape}, "
          f"sum={result['crack_mask'].sum()/255:.0f} 像素")
    print(f"  crack_keypoints:   {result['crack_keypoints'].shape}")
    print(f"  crack_bbox:        {result['crack_bbox']}")
    print(f"  peak_stress_MPa:   {result['metadata']['peak_stress_MPa']:.1f}")
    print(f"  total_time_h:      {result['metadata']['total_time_h']:.2f}")

    assert result['crack_mask'].sum() > 0, "应至少有部分裂纹像素"
    assert 8 <= len(result['crack_keypoints']) <= 16
    assert result['metadata']['peak_stress_MPa'] > 0
    print("\n[All Pass] PatchSimulator 端到端通过")