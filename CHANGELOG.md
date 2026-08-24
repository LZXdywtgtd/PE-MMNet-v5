# 更新日志 (CHANGELOG) — v5

> v5 独立版本线。从 v4.6.12 fork。
>
> 格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [v5.0.1-alpha] - 2026-08-12

### 新增（V5A1）

patch 仿真器升级（S1–S4 + S7）：

- `data/thermal_profile.py`（S4）：三段升降温曲线生成器
- `data/surface_radiation.py`（S2）：Stefan-Boltzmann 表面辐射
- `data/viscoelastic.py`（S3）：Maxwell 粘弹性应力模型
- `data/patch_simulator_v5.py`（S1 + S7 + 集成）：主仿真器
- `tools/patch_simulator_visualizer.py`：4 联可视化
- `tests/test_patch_simulator_v5.py`：12 项单元测试（全过）

### 物理修复

Maxwell update_step 公式多了一个 `(dt/τ)` 因子，导致累计应力被压缩4 个数量级（应 360 MPa，实际 0.02 MPa）。修正为标准离散化形式：

```
旧：σ_next = σ + dt·[(E/τ)·α·ΔT − σ/τ]
新：σ_next = σ + E·α·ΔT − (dt/τ)·σ
```

### 验证

- 单元测试：12/12 PASSED
- 端到端冒烟：64×64 patch, peak_σ = 59.2 MPa, K = 8 关键点
- 可视化：output/v5_sim_sample.png + v5_stress_evolution.png

### 已知问题新增

- V5-011：低阈值下 crack_mask 占满 patch（88% 覆盖率）
- V5-012：v5-α 裂纹萌生判据是简化版（未用 Griffith S5 + LEFM S6）
- V5-013：Maxwell 显式 Euler 在 dt > τ/2 时 clip 到 0

---

## [v5.0.0-alpha] - 2026-08-12

### Fork

- **基线版本**：v4.6.12
- **Fork 范围**：完整复制 `project_v4/` 到 `project_v5/`（跳过 checkpoints/logs/__pycache__/.git/benchmark_results）
- **目录关系**：v5 与 v4 并列（同级），v4 不动

### 范围声明

**v5-α 严格实现范围**：
- ✅ 256×256 虚拟 patch 输入（形状无关）
- ✅ 3 模态融合（温度场 + 应力场 + 热力图）
- ✅ 5 模型变体保留（resnet18 / swin_yolo / vit_yolo / detr / swin_yolo_patchtst）
- ✅ 8–16 动态有序关键点 + bbox 输出
- ✅ B 样条 + Hausdorff 覆盖距离损失
- ✅ 高斯过程嵌入模块（GP module）
- ✅ 泊松过程先验
- ✅ patch 仿真器升级（S1–S4：多方向热交换 + 辐射 + 应力松弛 + 三段曲线）
- ✅ 像素级裂纹 mask 输出（GT 配套）

**v5-α 严格不实现范围**（v5-β 文档占位）：
- ❌ 真实 3D 物体（杯/碗/盘等）
- ❌ UV 展开 + 多角度拍摄管线
- ❌ 真实烧制数据采集流程
- ❌ 位置图像 L0/L1 完整角点检测管线
- ❌ crack_annotator 标注工具
- ❌ 3D FEM 仿真

详细见 [docs/v5_范围说明.md](docs/v5_范围说明.md)。

### 任务清单（v5-α）

| 任务 | 名称 | 状态 |
|---|---|---|
| V5A1 | patch 仿真器升级（S1–S4） | 待启动 |
| V5A2 | Spline 输出头 + GP 模块 | 待启动 |
| V5A3 | 覆盖距离 loss | 待启动 |
| V5A4 | 5 变体输出头迁移 | 待 V5A2, V5A3 |
| V5A5 | 变长训练循环 | 待 V5A4 |
| V5A6 | 5 变体训练 + 评估 | 待 V5A1, V5A5 |

详细见 [tasks/team_v5_alpha.json](tasks/team_v5_alpha.json)。

### 已知问题

见 [docs/v5_已知问题.md](docs/v5_已知问题.md)。

---

## 迁移指南

### 从 v4 升级到 v5-α

**目录关系**：`project_v5/` 与 `project_v4/` 并列，不互相依赖。

**配置**：v5 使用独立 `config.json`，新增 `patch_size`、`kiln_dims`、`min_keypoints`、`max_keypoints` 等字段。

**数据**：v5-α 使用仿真生成的 patch 数据，无需 v4 真实数据。

**训练命令**（规划）：
```bash
cd D:\team_project\projects\pe_mmnet\project_v5
python run_train_v5.py --variant resnet18 --epochs 150 --min_kpts 8 --max_kpts 16
```

---

## 未来版本

### [v5.1.0] - v5-β（未规划时间）

- 真实 3D 物体接入
- UV 展开管线
- 多角度拍摄与 crack_annotator 工具
- 位置图像完整管线（L0/L1）
- 3D FEM 仿真
- 真实烧制数据采集 + GT 标注