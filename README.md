# PE-MMNet v5 多模态裂纹预测系统

> **Physics-Enhanced Multi-Modal Network v5**
>
> 基于深度学习的日用陶瓷热震裂纹**轨迹**实时预测系统
>
> 版本：5.0.7-alpha | 更新：2026-08-25
>
> v5 系列分两阶段：**v5-α**（MVP，算法骨架验证）+ **v5-β**（真实 3D 数据接入）

---

## ⚠ 范围声明

| 阶段 | 范围 | 状态 |
|---|---|---|
| **v5-α**（当前）| 256×256 虚拟 patch，3 模态输入，5 变体保留，**8–16 动态关键点 + bbox 输出** | V5A1-V5A5 ✅，V5A6 ⏳ |
| **v5-β**（启动条件：V5A6 通过）| 真实 3D 物体（杯/碗/盘）+ UV 展开 + 多角度拍摄 + 真实 GT 标注 + team_train + Streamlit | **部分已填实**：V5B3/V5B4/V5B6/V5B7 ✅（V5-β 钩子已预留）|
| **v5.1.0**（v5-β 完成后）| 生产级：HuggingFace 发布 + Web API + CI/CD | 远期 |

详细见 [docs/v5_范围说明.md](docs/v5_范围说明.md)。

**v5-α 不实现的内容**（v5-β 才做）：3D 物体建模、UV 展开、真实烧制数据采集、位置图像 L0/L1 完整管线（cv2 缺失时降级到 L3）、Griffith/LEFM 完整实现、Streamlit Web GUI。
**v5-β 已部分填实**：V5B3（crack_annotator）/ V5B4（position_extractor）/ V5B6（npz 缓存）/ V5B7（team_train_v5）— 等待陶艺作坊真实数据接入即可启用。

---

## v5-α vs v4.6.12 算法差异

| 维度 | v4.6.12 | v5-α |
|---|---|---|
| **输出** | 6 维向量（含 l, w）或二值掩膜 | **8–16 动态关键点 + bbox**（变长有序轨迹）|
| **输入** | 温度时序 + 温度/应力场 | **温度场 + 应力场 + 热力图**（位置图像可选 L3）|
| **物理约束** | 单调性损失（MSE + L_mono） | **高斯过程嵌入模块** + 泊松过程先验 |
| **损失** | 多类损失（Mono/Dice/YOLO/DETR）| **Catmull-Rom 样条 + Hausdorff 覆盖距离**（主）+ GP/泊松（辅）|
| **场景** | 1cm² 2D 平面 | **256×256 虚拟 patch**（形状无关）|
| **变体** | 5 个（resnet18/swin_yolo/vit_yolo/detr/swin_yolo_patchtst）| **5 个保留**，换输出头 |
| **仿真** | 简化单向热传导 | **多方向热交换 + 辐射 + 应力松弛 + 三段曲线** |
| **目录关系** | `project_v4/` | **`project_v5/` 并列**（v4 不动）|

## v5-β vs v5-α 算法扩展

| 维度 | v5-α | v5-β |
|---|---|---|
| **数据** | 仿真 patch | **真实烧后图 + UV 映射** |
| **3D** | 不支持 | **UV 展开 + 真实 3D 物体** |
| **位置图像** | L3 默认关闭 | **L0-L4 完整管线** |
| **GT** | 仿真器生成 | **crack_annotator 人工标注** |
| **GP** | batch 内 L2 归一化 | **running stats**（修复 V5-014/015）|
| **Griffith** | 简化 | **完整**（含缺陷尺寸）|
| **LEFM** | 简化 | **完整**（最大周向应力方向）|
| **数据缓存** | 内存 | **磁盘 hdf5/npz** |
| **团队协作** | 单任务 | **team_train_v5 + Streamlit** |

详细见 [docs/v5_架构设计.md §8](docs/v5_架构设计.md)。

---

## v5-α 任务清单（V5A1-V5A6）

| 任务 | 名称 | 优先级 | 依赖 | 状态 |
|---|---|---|---|---|
| **V5A1** | patch 仿真器升级（S1–S4 + S7） | P0 | — | ✅ 完成 (2026-08-12) |
| **V5A2** | Spline 输出头 + GP 模块 | P0 | — | ✅ 完成 (2026-08-12) |
| **V5A3** | 覆盖距离 loss（Catmull-Rom + Hausdorff） | P0 | — | ✅ 完成 (2026-08-12) |
| **V5A4** | 5 变体输出头迁移 | P0 | V5A2, V5A3 | ✅ 完成 (2026-08-12) |
| **V5A5** | 变长训练循环（trainer_v5） | P0 | V5A4 | ✅ 完成 (2026-08-12) |
| V5A6 | 5 变体训练 + 评估 | P0 | V5A1, V5A5 | 待启动 |

详细见 [tasks/team_v5_alpha.json](tasks/team_v5_alpha.json)。

---

## v5-β 任务清单（V5B1-V5B8，4 项已填实）

| 任务 | 名称 | 依赖 | 状态 |
|---|---|---|---|
| **V5B1** | UV 展开管线（3D mesh → 2D patch + 逆映射）| — | ⏳ 待启动 |
| **V5B2** | 多角度拍摄管线（窑内多机位同步采图）| V5B1 | ⏳ 待启动 |
| **V5B3** | crack_annotator 标注工具 | V5B2 | ✅ **已填实**（auto_extract + GUI + CLI） |
| **V5B4** | 位置图像 L0/L1 完整管线（角点检测 + 单应性）| V5B2 | ✅ **已填实**（5 级降级 L0-L4，L3/L4 完整，L0/L1 cv2 依赖）|
| **V5B5** | 3D FEM 仿真（可选）| V5A1 | ⏳ 待启动 |
| **V5B6** | 数据缓存 hdf5/npz（替换 PatchDatasetV5 内存 cache）| — | ✅ **已填实**（npz 实现，h5py 未装）|
| **V5B7** | team_train_v5.py（v4 移植 + v5-α 接口适配）| — | ✅ **已填实**（V5A*/V5B* 任务路由）|
| **V5B8** | Streamlit v5 推理 GUI（v5-α 占位升级）| V5B1, V5B3, V5B4 | ⏳ 待启动（当前用 matplotlib GUI 替代）|

**填实进度**：4/8 = 50%（V5B3/B4/B6/B7 已可用，剩余 V5B1/B2/B5/B8 依赖陶艺作坊合作）。
**代码重构**：TrainerV5 / OrderedKeypointLoss / PatchDatasetV5 全部已加 V5-β 钩子（默认值禁用，零 v5-α 影响）。

详见 [docs/团队协作训练指南.md §四](docs/团队协作训练指南.md) + [docs/api.md §8](docs/api.md)。

---

## 仿真升级清单（v5-α 完成 + v5-β 规划）

| 编号 | 升级项 | 来源 | 优先级 | v5-α 状态 | v5-β 状态 |
|---|---|---|---|---|---|
| S1 | 多方向热交换 | 会议 K9 | P0 | ✅ V5A1 | — |
| S2 | 表面辐射（Stefan-Boltzmann） | 物理约束 | P0 | ✅ V5A1 | — |
| S3 | 内部应力松弛 | 物理约束 | P0 | ✅ V5A1 | — |
| S4 | 分时段升降温曲线 | 会议 5 | P0 | ✅ V5A1 | — |
| S5 | 裂纹萌生判据（Griffith） | 物理约束 | P1 | 简化版 | **完整版** |
| S6 | 裂纹扩展轨迹生成（LEFM） | 物理约束 | P1 | 简化版 | **完整版** |
| S7 | 像素级裂纹 mask 输出 | GT 配套 | P0 | ✅ V5A1 | — |
| 3D FEM | 3D 有限元仿真 | 物理约束 | P2 | ❌ | **可选** |

详细见 [docs/v5_仿真升级清单.md](docs/v5_仿真升级清单.md)。

---

## 文档入口（全部平铺到 docs/ 根）

| 类别 | 文档 |
|---|---|
| **范围（α + β）** | [v5_范围说明.md](docs/v5_范围说明.md) |
| **架构（α + β）** | [v5_架构设计.md](docs/v5_架构设计.md) |
| **仿真（α + β）** | [v5_仿真升级清单.md](docs/v5_仿真升级清单.md) |
| **已知问题** | [v5_已知问题.md](docs/v5_已知问题.md)（V5-001 至 V5-020 + v4 B001-B013 继承）|
| **算法报告（α + β）** | [项目算法与训练实验设计报告.md](docs/项目算法与训练实验设计报告.md) |
| **调参** | [调参与算法工程指导文档.md](docs/调参与算法工程指导文档.md) |
| **开发** | [开发人员文档.md](docs/开发人员文档.md) |
| **API** | [api.md](docs/api.md) |
| **快速上手** | [快速配置指南.md](docs/快速配置指南.md) |
| **CUDA** | [CUDA安装指南.md](docs/CUDA安装指南.md) |
| **团队协作** | [团队协作训练指南.md](docs/团队协作训练指南.md)（V5B7 已实现）|
| **流程** | [全流程.md](docs/全流程.md) |
| **文档审查** | [文档审查指南.md](docs/文档审查指南.md) |

> v5-α 文档**完全独立**，**无需查阅 v4 docs**。每个文档顶部"继承说明"段注明与 v4 的关系。

---

## 快速开始

### v5-α 冒烟测试（1 min）

```bash
# 1. 激活 conda 环境
conda activate pe_mmnet

# 2. 冒烟测试
cd D:\team_project\projects\pe_mmnet\project_v5
python run_train_v5.py --variant resnet18 --epochs 1 --n_samples 2 \
    --patch_size 64 --fast_thermal --log_dir logs/smoke
```

### v5-α 完整训练（150 epoch）

```bash
# 单变体 150 epoch × 默认 11.7h 仿真周期
python run_train_v5.py --variant resnet18 --epochs 150 --n_samples 1000 \
    --log_dir logs/training_history/v5a6_resnet18

# 跑全部 5 变体（V5A6）
for v in resnet18 swin_yolo vit_yolo detr swin_yolo_patchtst; do
    python run_train_v5.py --variant $v --epochs 150 --n_samples 1000 \
        --log_dir logs/training_history/v5a6_$v
done
```

### v5-β（待启动）

启动条件：V5A6 评估报告显示 5 变体均跑通且与 v4 baseline 对比有提升。

详见 [docs/v5_范围说明.md §二](docs/v5_范围说明.md) + [docs/v5_架构设计.md §8](docs/v5_架构设计.md)。

详见 [docs/快速配置指南.md](docs/快速配置指南.md)。

---

## 测试覆盖

| 阶段 | 项数 | 文件 |
|---|---|---|
| V5A1 | 12 | `tests/test_patch_simulator_v5.py` |
| V5A2 | 18 | `tests/test_spline_head.py` + `test_gp_module.py` |
| V5A3 | 24 | `tests/test_ordered_kp_loss.py` |
| V5A4 | 18 | `tests/test_pe_tsnet_v5.py` |
| V5A5 | 17 | `tests/test_trainer_v5_smoke.py` |
| V5B 钩子 | 12 | `tests/test_v5_beta_hooks.py` |
| **合计** | **101/101 PASSED** | |

v5-β 启动后将新增对应测试（V5B1 UV 展开、V5B2 多机位、V5B5 3D FEM、V5B8 Streamlit）。

---

## 更新日志

详见 [CHANGELOG.md](CHANGELOG.md)。v5 从 v5.0.0 起独立版本线，v5-β 计划 v5.1.0。

---

## 维护说明

v5-α 处于 MVP 阶段，每完成一项任务（V5A1–V5A6）走"修复 → 审计 → 更新文档 → 推送"循环（详见 [docs/全流程.md](docs/全流程.md)）。