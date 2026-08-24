# PE-MMNet v5 多模态裂纹预测系统

> **Physics-Enhanced Multi-Modal Network v5**
>
> 基于深度学习的日用陶瓷热震裂纹**轨迹**实时预测系统
>
> 版本：5.0.4-alpha | 更新：2026-08-12 | Fork 自 v4.6.12

---

## ⚠️ 范围声明（v5-α MVP）

v5-α 范围**严格限定**：

| 范围 | 内容 |
|---|---|
| **v5-α（当前实施）** | 256×256 虚拟 patch，3 模态输入，5 变体保留，**8–16 动态关键点 + bbox 输出** |
| **v5-β（文档占位，未实现）** | 真实 3D 物体（杯/碗/盘）+ UV 展开 + 多角度拍摄 + 真实 GT 标注 |

**v5-α 不实现的内容**：3D 物体建模、UV 展开、真实烧制数据采集、位置图像 L0/L1 完整管线、crack_annotator 标注工具。

详细见 [docs/v5_范围说明.md](docs/v5_范围说明.md)。

---

## 与 v4 的核心差异

| 维度 | v4.6.12 | v5-α |
|---|---|---|
| **输出** | 6 维向量（含 l, w）或二值掩膜 | **8–16 动态关键点 + bbox**（变长有序轨迹） |
| **输入** | 温度时序 + 温度/应力场 | **温度场 + 应力场 + 热力图**（位置图像可选 L3） |
| **物理约束** | 单调性损失（MSE + L_mono） | **高斯过程嵌入模块** + 泊松过程先验 |
| **损失** | 多类损失（Mono/Dice/YOLO/DETR） | **B 样条 + Hausdorff 覆盖距离**（主）+ GP/泊松（辅） |
| **场景** | 1cm² 2D 平面 | **256×256 虚拟 patch**（形状无关） |
| **变体** | 5 个（resnet18/swin_yolo/vit_yolo/detr/swin_yolo_patchtst） | **5 个保留**，换输出头 |
| **仿真** | 简化单向热传导 | **多方向热交换 + 辐射 + 应力松弛 + 三段曲线** |
| **目录关系** | `project_v4/` | **`project_v5/` 并列**（v4 不动） |

详细见 [docs/v5_架构设计.md](docs/v5_架构设计.md)。

---

## v5-α 任务清单

| 任务 | 名称 | 优先级 | 依赖 | 状态 |
|---|---|---|---|---|
| **V5A1** | patch 仿真器升级（S1–S4 + S7） | P0 | — | ✅ 完成 (2026-08-12) |
| **V5A2** | Spline 输出头 + GP 模块 | P0 | — | ✅ 完成 (2026-08-12) |
| **V5A3** | 覆盖距离 loss（B 样条 + Hausdorff） | P0 | — | ✅ 完成 (2026-08-12) |
| **V5A4** | 5 变体输出头迁移 | P0 | V5A2, V5A3 | ✅ 完成 (2026-08-12) |
| V5A5 | 变长训练循环 | P0 | V5A4 | 待 V5A4 |
| V5A6 | 5 变体训练 + 评估 | P0 | V5A1, V5A5 | 待 V5A5 |

详细见 [tasks/team_v5_alpha.json](tasks/team_v5_alpha.json)。

---

## 仿真升级清单

| 编号 | 升级项 | 来源 | 优先级 |
|---|---|---|---|
| S1 | 多方向热交换 | 会议 K9 | P0 |
| S2 | 表面辐射（Stefan-Boltzmann） | 物理约束 | P0 |
| S3 | 内部应力松弛 | 物理约束 | P0 |
| S4 | 分时段升降温曲线 | 会议 5 | P0 |
| S5 | 裂纹萌生判据（Griffith） | 物理约束 | P1 |
| S6 | 裂纹扩展轨迹生成（LEFM） | 物理约束 | P1 |
| S7 | 像素级裂纹 mask 输出 | GT 配套 | P0 |

详细见 [docs/v5_仿真升级清单.md](docs/v5_仿真升级清单.md)。

---

## 文档入口

- 范围声明 → [docs/v5_范围说明.md](docs/v5_范围说明.md)
- 架构设计 → [docs/v5_架构设计.md](docs/v5_架构设计.md)
- 仿真升级 → [docs/v5_仿真升级清单.md](docs/v5_仿真升级清单.md)
- 已知问题 → [docs/v5_已知问题.md](docs/v5_已知问题.md)
- 文档地图 → [docs/文档审查指南.md](docs/文档审查指南.md)
- 流程循环 → [docs/全流程.md](docs/全流程.md)

---

## v4 沿用（v5 fork 自 v4，未改动）

- 用户指南 → [docs/user_guides/](docs/user_guides/)
- 开发参考 → [docs/dev_reference/](docs/dev_reference/)
- 协作训练 → [docs/collaboration/](docs/collaboration/)
- 实验报告 → [docs/experiment_reports/](docs/experiment_reports/)

---

## 更新日志

详见 [CHANGELOG.md](CHANGELOG.md)。v5 从 v5.0.0 起独立版本线。

---

## 维护说明

v5-α 处于 MVP 阶段，每完成一项任务（V5A1–V5A6）走"修复 → 审计 → 更新文档 → 推送"循环（详见 [docs/全流程.md](docs/全流程.md)）。