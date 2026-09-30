"""
v5-α 训练循环（V5A5 + V5-β 钩子）

TrainerV5：变长关键点训练循环，断点续训，CSV 历史，GPU 自适应。

核心功能：
1. 变长 batch（collate_v5 已处理 padding + mask）
2. GP 模块正则注入（OrderedKeypointLoss 内置）
3. 覆盖距离 loss 调用（OrderedKeypointLoss 内置）
4. 动态 K 选择（SplineHead 内置）
5. 断点续训（save/load_state）
6. CSV 训练历史
7. 评估模式（coverage/bbox/KP IoU）

[v5-β 钩子]（v5-α 全部默认 False/None，不影响当前行为）：
- data_source: "simulation" / "real" —— 数据源切换（V5B6 接入真实数据）
- use_position_image: bool —— 是否注入位置图像分支（V5B4）
- use_uv_inverse: bool —— 是否注入 UV 逆映射 head（V5B1）
- cache_dir: str | None —— 仿真 npz 缓存目录（V5B6 磁盘缓存）

详见：
- docs/v5_架构设计.md §4.5 + §8 v5-β 架构
- tasks/team_v5_alpha.json 中 V5A5
- docs/v5_范围说明.md §11 v5-β 8 任务
"""

from __future__ import annotations

import os
import csv
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import create_v5_model
from training.ordered_kp_loss import OrderedKeypointLoss
from training.data_aug_v5 import PatchAugmentorV5
from data.patch_dataset_v5 import PatchDatasetV5, collate_v5


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """获取设备（默认 CUDA 可用就用）"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
#  CSV 历史
# ============================================================

class CSVHistory:
    """
    训练历史 CSV 记录器

    列：epoch, train_loss, val_loss, lr, epoch_time_s, timestamp
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.fieldnames = [
            "epoch", "train_loss", "train_coverage", "train_bbox",
            "train_gp", "train_poisson",
            "train_position_6d", "train_uv_projection",  # v5-β 钩子列
            "val_loss", "val_coverage", "val_pos_batch_frac",
            "lr", "epoch_time_s", "timestamp",
        ]
        # 文件不存在则写表头
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: dict) -> None:
        """写入一行"""
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


# ============================================================
#  检查点管理
# ============================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    history_csv: str,
    extras: dict | None = None,
) -> None:
    """保存训练检查点"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "history_csv": history_csv,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().tolist(),
        },
    }
    if extras:
        state.update(extras)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """加载训练检查点，返回状态字典（含 epoch, best_val_loss 等）"""
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    # 恢复 RNG
    if "rng_state" in state:
        try:
            random.setstate(state["rng_state"]["python"])
            np.random.set_state(state["rng_state"]["numpy"])
            torch_state = state["rng_state"]["torch"]
            if isinstance(torch_state, list):
                torch.set_rng_state(torch.tensor(torch_state, dtype=torch.uint8))
            else:
                torch.set_rng_state(torch_state)
        except Exception as e:
            print(f"  [WARN] RNG state 恢复失败: {e}")
    return state


# ============================================================
#  Trainer
# ============================================================

class TrainerV5:
    """
    v5-α 训练器（V5-β 钩子已预留）

    Args:
        variant: 模型变体（"resnet18" / "swin_yolo" / "vit_yolo" / "detr" / "swin_yolo_patchtst"）
        n_samples: 数据集样本数
        patch_size: patch 像素尺寸
        batch_size: 批大小
        epochs: 总 epoch 数
        lr: 学习率
        min_kpts / max_kpts: 关键点数范围
        coverage_mode: "chamfer" / "hausdorff"
        M: 样条采样点数
        lambda_coverage / lambda_bbox / lambda_gp / lambda_poisson: loss 权重
        use_aug: 是否启用数据增强
        seed: 随机种子
        device: "cuda" / "cpu"
        log_dir: 日志目录（CSV + checkpoints）
        thermal_profile: 传给 PatchDatasetV5 的热曲线配置（None=默认）
        crack_stress_threshold_MPa: 裂纹阈值（默认 50.0）
        save_every: 每 N epoch 保存一次 checkpoint
        model_kwargs: 传给 create_v5_model 的额外 kwargs
        ---- v5-β 钩子（默认 None/False，v5-α 完全不启用） ----
        data_source: "simulation"（v5-α 默认）/ "real"（v5-β 真实数据）
        cache_dir: 仿真 npz 磁盘缓存目录（v5-β V5B6）
        use_position_image: 是否注入位置图像分支（v5-β V5B4）
        use_uv_inverse: 是否注入 UV 逆映射（v5-β V5B1）
        lambda_position_6d: 位置 6D loss 权重（v5-β）
        lambda_uv_projection: UV 投影 loss 权重（v5-β）
    """

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
        coverage_mode: str = "chamfer",
        M: int = 200,
        lambda_coverage: float = 1.0,
        lambda_bbox: float = 1.0,
        lambda_gp: float = 0.1,
        lambda_poisson: float = 0.05,
        lambda_kpt_direct: float = 0.5,
        lambda_ordered: float = 0.0,
        lambda_validity: float = 0.0,
        lambda_heatmap: float = 0.0,
        spatial_head: bool = False,
        use_aug: bool = True,
        seed: int = 42,
        device: str | None = None,
        log_dir: str = "logs/training_history/v5a5_smoke",
        thermal_profile: dict | None = None,
        crack_stress_threshold_MPa: float = 50.0,
        save_every: int = 1,
        model_kwargs: dict | None = None,
        # ---- v5-β 钩子（默认禁用） ----
        data_source: str = "simulation",
        cache_dir: str | None = None,
        use_position_image: bool = False,
        use_uv_inverse: bool = False,
        lambda_position_6d: float = 0.0,
        lambda_uv_projection: float = 0.0,
    ):
        self.variant = variant
        self.n_samples = n_samples
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.min_kpts = min_kpts
        self.max_kpts = max_kpts
        self.coverage_mode = coverage_mode
        self.M = M
        self.use_aug = use_aug
        self.seed = seed
        self.log_dir = log_dir
        self.save_every = save_every
        self.model_kwargs = model_kwargs or {}

        # ---- v5-β 钩子 ----
        self.data_source = data_source
        self.cache_dir = cache_dir
        self.use_position_image = use_position_image
        self.use_uv_inverse = use_uv_inverse
        self.lambda_position_6d = lambda_position_6d
        self.lambda_uv_projection = lambda_uv_projection

        # V5-β 钩子守卫：仅记录意图，不抛错（让 v5-α 用户无感）
        if data_source not in ("simulation", "real"):
            raise ValueError(
                f"data_source 应为 'simulation' / 'real'，实际 {data_source!r}"
            )
        if use_position_image and data_source != "real":
            import warnings
            warnings.warn(
                "use_position_image=True 但 data_source='simulation'。"
                "位置图像分支将注入全零 + mask（v5-α 仿真数据无 6D pose 信息）。",
                UserWarning,
                stacklevel=2,
            )
        if use_uv_inverse and data_source != "real":
            import warnings
            warnings.warn(
                "use_uv_inverse=True 但 data_source='simulation'。"
                "UV 逆映射将退化为 identity（v5-α 无 UV 网格）。",
                UserWarning,
                stacklevel=2,
            )

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

        # 设备
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        # 随机种子
        set_seed(seed)

        # 数据集（v5-β: 支持 npz 缓存 + position_6d 字段）
        dataset_kwargs = dict(
            patch_size=patch_size,
            n_samples=n_samples,
            min_kpts=min_kpts,
            max_kpts=max_kpts,
            seed=seed,
            thermal_profile=thermal_profile,
            crack_stress_threshold_MPa=crack_stress_threshold_MPa,
            data_source=data_source,
            use_position_image=use_position_image,
            cache_dir=cache_dir,
        )
        self.train_ds = PatchDatasetV5(**dataset_kwargs)
        # 验证集：同 seed 偏移（数据不同）
        self.val_ds = PatchDatasetV5(
            **dict(dataset_kwargs, n_samples=max(2, n_samples // 5), seed=seed + 10000),
        )

        # 增强器
        if use_aug:
            self.augmentor = PatchAugmentorV5()
        else:
            self.augmentor = None

        # DataLoader（num_workers=0 因为 dataset 已预计算）
        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_v5, drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_v5, drop_last=False,
        )

        # 模型
        # 合并 model_kwargs 和基础参数（避免冲突）
        self.use_spatial_head = bool(spatial_head)
        self.lambda_heatmap = lambda_heatmap
        build_kwargs = {
            "image_channels": 3,
            "image_size": patch_size,
            "pretrained_2d": self.model_kwargs.pop("pretrained_2d", False),
            "use_gp": True,
            "max_kpts": max_kpts,
            "min_kpts": min_kpts,
            **self.model_kwargs,
        }
        # spatial_head 仅 resnet18（PETSNetMultimodalV5）支持，其余变体不传
        if self.use_spatial_head:
            build_kwargs["spatial_head"] = True
        self.model = create_v5_model(variant, **build_kwargs).to(self.device)

        # 损失（v5-β: 增加 position_6d_loss / uv_projection_loss 占位）
        self.loss_fn = OrderedKeypointLoss(
            lambda_coverage=lambda_coverage,
            lambda_bbox=lambda_bbox,
            lambda_gp=lambda_gp,
            lambda_poisson=lambda_poisson,
            lambda_kpt_direct=lambda_kpt_direct,
            lambda_ordered=lambda_ordered,
            lambda_validity=lambda_validity,
            coverage_mode=coverage_mode,
            M=M,
            lambda_position_6d=lambda_position_6d,
            lambda_uv_projection=lambda_uv_projection,
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # CSV
        self.csv_path = os.path.join(log_dir, "training_history.csv")
        self.history = CSVHistory(self.csv_path)

        # 训练状态
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_coverage = float("inf")

    def _train_step(self, batch: dict) -> dict:
        """单步训练"""
        x_1d = batch["x_1d"].to(self.device)
        x_2d = batch["x_2d"].to(self.device)
        true_bbox = batch["true_bbox"].to(self.device)
        true_kpts = batch["true_keypoints"].to(self.device)
        true_pixels = batch["true_crack_pixels"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        kpt_mask = batch["keypoint_mask"].to(self.device)
        # v5-β 钩子：position_6d（若存在；否则 None）
        true_position_6d = None
        if self.use_position_image and "true_position_6d" in batch:
            true_position_6d = batch["true_position_6d"].to(self.device)

        # 数据增强（先转 CPU）
        if self.augmentor is not None:
            aug_x_1d, aug_x_2d, aug_bbox, aug_kpts, aug_pixels = [], [], [], [], []
            for i in range(x_1d.size(0)):
                a1, a2, ab, ak, ap = self.augmentor(
                    x_1d[i].cpu(), x_2d[i].cpu(),
                    true_bbox[i].cpu(),
                    true_kpts[i][kpt_mask[i]].cpu(),
                    true_pixels[i][pixel_mask[i]].cpu(),
                )
                aug_x_1d.append(a1)
                aug_x_2d.append(a2)
                aug_bbox.append(ab)
                aug_kpts.append(ak)
                aug_pixels.append(ap)
            x_1d = torch.stack(aug_x_1d).to(self.device)
            x_2d = torch.stack(aug_x_2d).to(self.device)
            true_bbox = torch.stack(aug_bbox).to(self.device)
            # aug_kpts / aug_pixels 长度可能不同，重新 pad
            max_k = max(k.size(0) for k in aug_kpts)
            true_kpts = torch.full((x_1d.size(0), max_k, 2), -1.0, device=self.device)
            kpt_mask = torch.zeros(x_1d.size(0), max_k, dtype=torch.bool, device=self.device)
            for i, k in enumerate(aug_kpts):
                true_kpts[i, :k.size(0)] = k.to(self.device)
                kpt_mask[i, :k.size(0)] = True

            max_n = max(p.size(0) for p in aug_pixels) if aug_pixels else 0
            if max_n > 0:
                true_pixels = torch.full((x_1d.size(0), max_n, 2), -1.0, device=self.device)
                pixel_mask = torch.zeros(x_1d.size(0), max_n, dtype=torch.bool, device=self.device)
                for i, p in enumerate(aug_pixels):
                    if p.size(0) > 0:
                        true_pixels[i, :p.size(0)] = p.to(self.device)
                        pixel_mask[i, :p.size(0)] = True
            else:
                true_pixels = torch.zeros(x_1d.size(0), 0, 2, device=self.device)
                pixel_mask = torch.zeros(x_1d.size(0), 0, dtype=torch.bool, device=self.device)

        self.optimizer.zero_grad()
        out = self.model(x_1d, x_2d)

        # 计算 loss（GP 模块从模型取）
        gp_module = getattr(self.model, "gp_module", None)
        losses = self.loss_fn(
            pred_bbox=out["bbox"],
            pred_kpts=out["keypoints"],
            true_bbox=true_bbox,
            true_crack_pixels=true_pixels,
            gp_module=gp_module,
            pixel_mask=pixel_mask,
            true_kpts=true_kpts,
            kpt_mask=kpt_mask,
            pred_validity=out.get("validity"),
            true_position_6d=true_position_6d,
            pred_position_6d=out.get("position_6d"),
        )
        # heatmap 辅助监督并入 total（必须在 step() 之前 backward，
        # 否则梯度被下轮 zero_grad 清掉、监督无效——V5-030 第三轮）
        heatmap_val = 0.0
        hm = out.get("heatmap")
        if hm is not None and self.lambda_heatmap > 0:
            from models.spatial_kpt_head import heatmap_loss
            hm_t = heatmap_loss(hm, pixel_mask, true_pixels)
            losses["total"] = losses["total"] + self.lambda_heatmap * hm_t
            heatmap_val = hm_t.item()

        losses["total"].backward()
        self.optimizer.step()

        return {
            "loss": losses["total"].item(),
            "coverage": losses["coverage"].item(),
            "ordered": losses["ordered"].item(),
            "validity": losses["validity"].item(),
            "heatmap": heatmap_val,
            "bbox": losses["bbox"].item(),
            "gp": losses["gp"].item(),
            "poisson": losses["poisson"].item(),
            "position_6d": losses["position_6d"].item() if losses["position_6d"].numel() else 0.0,
            "uv_projection": losses["uv_projection"].item() if losses["uv_projection"].numel() else 0.0,
        }

    @torch.no_grad()
    def _eval_step(self, batch: dict) -> dict:
        """单步评估（无增强）"""
        x_1d = batch["x_1d"].to(self.device)
        x_2d = batch["x_2d"].to(self.device)
        true_bbox = batch["true_bbox"].to(self.device)
        true_pixels = batch["true_crack_pixels"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        true_kpts = batch["true_keypoints"].to(self.device)
        kpt_mask = batch["keypoint_mask"].to(self.device)

        out = self.model(x_1d, x_2d)
        gp_module = getattr(self.model, "gp_module", None)
        # v5-β 钩子
        true_position_6d = None
        if self.use_position_image and "true_position_6d" in batch:
            true_position_6d = batch["true_position_6d"].to(self.device)
        losses = self.loss_fn(
            pred_bbox=out["bbox"],
            pred_kpts=out["keypoints"],
            true_bbox=true_bbox,
            true_crack_pixels=true_pixels,
            gp_module=gp_module,
            pixel_mask=pixel_mask,
            true_kpts=true_kpts,
            kpt_mask=kpt_mask,
            pred_validity=out.get("validity"),
            true_position_6d=true_position_6d,
            pred_position_6d=out.get("position_6d"),
        )
        # heatmap 辅助（spatial_head 时激活）
        hm_val = 0.0
        hm = out.get("heatmap")
        if hm is not None and self.lambda_heatmap > 0:
            from models.spatial_kpt_head import heatmap_loss
            hm_t = heatmap_loss(hm, pixel_mask, true_pixels)
            hm_val = hm_t.item()
            losses["total"] = losses["total"] + self.lambda_heatmap * hm_t
        return {
            "loss": losses["total"].item(),
            "coverage": losses["coverage"].item(),
            "n_pos_batch": 1 if bool(pixel_mask.any()) else 0,
            "heatmap": hm_val,
        }

    def train_epoch(self, epoch: int) -> dict:
        """训练一个 epoch"""
        self.model.train()
        agg = {
            "loss": 0.0, "coverage": 0.0, "bbox": 0.0,
            "gp": 0.0, "poisson": 0.0,
            "ordered": 0.0, "validity": 0.0, "heatmap": 0.0,
            "position_6d": 0.0, "uv_projection": 0.0,
        }
        n = 0
        for batch in self.train_loader:
            step = self._train_step(batch)
            for k in agg:
                agg[k] += step[k]
            n += 1
        for k in agg:
            agg[k] /= max(n, 1)
        return agg

    @torch.no_grad()
    def eval_epoch(self) -> dict:
        """评估一个 epoch"""
        self.model.eval()
        agg = {"loss": 0.0, "coverage": 0.0, "n_pos_batch": 0}
        n = 0
        for batch in self.val_loader:
            step = self._eval_step(batch)
            for k in agg:
                agg[k] += step[k]
            n += 1
        for k in agg:
            agg[k] /= max(n, 1)
        # V5-031 纪律 3：coverage_loss 全负 batch 返回 0 且计入均值，
        # val_coverage 被正样本率稀释（augfix1 事故：stock 200 仅 12 正，
        # 38/50 batch 贡献 0 → 0.0218 假低值，_diag_stock6x.py 逐位复现）。
        # 主口径改为仅正样本 batch 均值；n_pos_batch 同步落 CSV 供核对。
        pos_frac = agg["n_pos_batch"]
        if pos_frac > 0:
            # agg["coverage"] 目前是"含零均值"；换算回仅正样本均值：
            # sum_cov = agg["coverage"] * n；pos_mean = sum_cov / (pos_frac)
            agg["coverage_pos"] = (agg["coverage"] * n) / pos_frac
        else:
            agg["coverage_pos"] = agg["coverage"]
        return agg

    def fit(
        self,
        resume_from: str | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        训练主循环

        Args:
            resume_from: 检查点路径（None = 全新训练）
            verbose: 是否打印进度

        Returns:
            final_metrics: dict
        """
        if resume_from is not None and os.path.exists(resume_from):
            state = load_checkpoint(
                resume_from, self.model, self.optimizer,
                map_location=self.device,
            )
            self.start_epoch = state.get("epoch", 0) + 1
            self.best_val_loss = state.get("best_val_loss", float("inf"))
            self.best_val_coverage = state.get(
                "best_val_coverage", float("inf")
            )
            if verbose:
                print(f"  [RESUME] 从 epoch {self.start_epoch} 续训，"
                      f"best_val_coverage={self.best_val_coverage:.4f}")

        final = {}
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.eval_epoch()
            dt = time.time() - t0

            row = {
                "epoch": epoch + 1,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_coverage": f"{train_metrics['coverage']:.6f}",
                "train_bbox": f"{train_metrics['bbox']:.6f}",
                "train_gp": f"{train_metrics['gp']:.6f}",
                "train_poisson": f"{train_metrics['poisson']:.6f}",
                "train_position_6d": f"{train_metrics.get('position_6d', 0.0):.6f}",
                "train_uv_projection": f"{train_metrics.get('uv_projection', 0.0):.6f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_coverage": f"{val_metrics['coverage_pos']:.6f}",
                "val_pos_batch_frac": f"{val_metrics['n_pos_batch']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                "epoch_time_s": f"{dt:.2f}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.history.log(row)

            # 保存 best checkpoint（V5-030：按 val_coverage 选——val_loss 含
            # bbox/gp 分量波动，曾把 best 停在 ep44 而 ep150 的 val_coverage 更低，
            # 判据 4 结论被此污染；coverage 才是判据 4 直接衡量的量。
            # V5-031：用 coverage_pos（仅正样本 batch 口径）——含零口径受
            # 正样本率稀释，best 选择会随数据集正样本率漂移）
            if val_metrics["coverage_pos"] < self.best_val_coverage:
                self.best_val_coverage = val_metrics["coverage_pos"]
                best_path = os.path.join(
                    self.log_dir, "checkpoints", "best.pt"
                )
                save_checkpoint(
                    best_path, self.model, self.optimizer,
                    epoch=epoch, best_val_loss=self.best_val_loss,
                    history_csv=self.csv_path,
                    extras={
                        "variant": self.variant,
                        "best_val_coverage": self.best_val_coverage,
                    },
                )

            # 每 save_every epoch 保存 latest
            if (epoch + 1) % self.save_every == 0:
                latest_path = os.path.join(
                    self.log_dir, "checkpoints", "latest.pt"
                )
                save_checkpoint(
                    latest_path, self.model, self.optimizer,
                    epoch=epoch, best_val_loss=self.best_val_loss,
                    history_csv=self.csv_path,
                    extras={
                        "variant": self.variant,
                        "best_val_coverage": self.best_val_coverage,
                    },
                )

            if verbose:
                print(
                    f"  [Epoch {epoch + 1:3d}/{self.epochs}] "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"cov={train_metrics['coverage']:.4f} "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e} "
                    f"time={dt:.1f}s"
                )

            final = {
                "last_train_loss": train_metrics["loss"],
                "last_val_loss": val_metrics["loss"],
                "best_val_loss": self.best_val_loss,
                "best_val_coverage": self.best_val_coverage,
            }

        return final


# ============================================================
#  冒烟测试
# ============================================================

def _smoke_test():
    print("=" * 60)
    print("TrainerV5 冒烟测试")
    print("=" * 60)

    import shutil
    smoke_log_dir = "logs/training_history/v5a5_smoke"
    if os.path.exists(smoke_log_dir):
        shutil.rmtree(smoke_log_dir)
    smoke_resume_dir = "logs/training_history/v5a5_smoke_resume"
    if os.path.exists(smoke_resume_dir):
        shutil.rmtree(smoke_resume_dir)

    # 用快速热曲线（1 min 周期）+ 小 patch
    fast_profile = {
        "ramp_up_c_per_min": 1260.0,
        "soak_temp_c": 1280.0,
        "soak_duration_min": 1.0,
        "cool_down_c_per_min": 1260.0,
    }

    trainer = TrainerV5(
        variant="resnet18",
        n_samples=2,         # 最小数据集
        patch_size=64,
        batch_size=2,
        epochs=2,
        lr=1e-3,
        crack_stress_threshold_MPa=5.0,  # 低阈值确保裂纹萌生
        log_dir=smoke_log_dir,
        thermal_profile=fast_profile,
        save_every=1,
    )

    print(f"\n[OK] Trainer 初始化: 模型={trainer.variant}, "
          f"参数={sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"     设备: {trainer.device}")
    print(f"     训练样本: {len(trainer.train_ds)}")
    print(f"     验证样本: {len(trainer.val_ds)}")

    final = trainer.fit(verbose=True)

    # 验证检查点存在
    best_pt = os.path.join(trainer.log_dir, "checkpoints", "best.pt")
    latest_pt = os.path.join(trainer.log_dir, "checkpoints", "latest.pt")
    assert os.path.exists(best_pt), f"best.pt 不  {best_pt}"
    assert os.path.exists(latest_pt), f"latest.pt 不  {latest_pt}"

    # 验证 CSV 存在 + 行数
    assert os.path.exists(trainer.csv_path), f"CSV 不  {trainer.csv_path}"
    with open(trainer.csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3, f"CSV 应 1 header + 2 epoch = 3 行，实际 {len(lines)}"

    # 验证断点续训
    print(f"\n[断点续训测试]")
    trainer2 = TrainerV5(
        variant="resnet18",
        n_samples=2,
        patch_size=64,
        batch_size=2,
        epochs=4,             # 共 4 epoch，第 2 个 epoch 续训
        lr=1e-3,
        crack_stress_threshold_MPa=5.0,
        log_dir=smoke_resume_dir,
        thermal_profile=fast_profile,
        save_every=1,
    )
    trainer2.start_epoch = 2  # 模拟从 epoch 2 续训
    trainer2.best_val_loss = 0.5
    state_path = latest_pt
    if os.path.exists(state_path):
        loaded = load_checkpoint(state_path, trainer2.model, trainer2.optimizer)
        assert loaded["epoch"] == 1, f"应加载 epoch=1，实际 {loaded['epoch']}"
        print(f"  [OK] 加载 latest.pt: epoch={loaded['epoch']}, "
              f"best_val_loss={loaded['best_val_loss']:.4f}")

    print(f"\n[ALL PASS] ✅")


if __name__ == "__main__":
    _smoke_test()