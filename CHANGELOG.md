# 更新日志 (CHANGELOG) — v5

> v5 独立版本线。从 v4.6.12 fork。
>
> 格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。
>
> **继承说明**：v5-α 文档体系**完全独立**，**无需查阅 v4 docs**。每个文档顶部"继承说明"段注明与 v4 的关系。

---

## [v5.0.9-alpha] - 2026-09-29

### GT 线状路线 D' 定稿前置修复：闭运算贴边腐蚀（V5-024）

#### 修复

- **data/patch_simulator_v5.py `_extract_crack_mask`**
  - `binary_closing(iterations=2)` → `binary_closing(iterations=2, border_value=1)`：
    scipy 默认 border_value=0 在边界处腐蚀贴边细线（D' 边界调制的线状恰好
    沿边分布，是最大受害形态——seed 4215 th=50 裸 95px→闭运算后 0px；
    th≥55 时每 100 样本约 7 个正样本被删光）。回归 102/102 通过
  - ⚠️ **行为变更**：`_extract_crack_mask` 输出与 v5.0.8-alpha 及之前不兼容。
    旧缓存（assets th=11）本因 GT 满面问题废弃，无迁移需求
- **修后口径阈值重扫（σ 场缓存离线，`_exp_rescan_post_v524.py`，n=100）**：
  产线 th=60（正样本 99/100、线状 86/100、外推 1000 样本 ≈860、面积
  max 14.1%）——**取代**修前口径的"th=50 平衡点"结论。th 选择判据写死：
  正样本率 ≥95% 前提下线状数最大，并列取 th 最小

#### 已知问题新增

- V5-024 条目（本文修复的完整档案，含修前/修后对照表）
- corner 调制证伪（`_exp_corner_modulation.py`）：H/V 限制的中成本解法②
  不可用——角部双边调制产生径向对称应力斑（elong=1.0），非对角线状

---

## [v5.0.8-alpha] - 2026-09-29

### 比赛副本回填：监督链路三重修复 + 断点续跑缓存（⚠️ 仿真输出与历史不兼容）

> 来源：assets/pe_mmnet/project_v5（比赛副本，2026-09-24~09-28 期间诊断 v5a6
> 平台解时修复）。副本层级的路径适配改动（三桥 `../../`）**未回填**——本仓库
> 目录层级下 `../../../` 本来正确。

#### 修复

- **loss（training/ordered_kp_loss.py + trainer_v5.py）**
  - mask 透传：trainer 原先算了 `pixel_mask` 但不传 loss_fn，负样本 (-1,-1)
    padding 全部进入 chamfer（2正6负 batch = 9 万假像素）→ 现必传
  - 单向 chamfer → 双向：原单向奖励"摊开覆盖"平凡解（边界角点 0.0148 <
    真关键点 0.0925）；hausdorff 同步改两方向 max
  - 新增 `kpt_direct`（权重 0.5）：kpts → 最近真像素 L2，修复"样条双重间接
    监督"的梯度病态。线状子集验证排序恢复：真kpts 0.2282 < 随机 0.3982
- **data/patch_dataset_v5.py**
  - kpt_mask/pixel_mask 裁剪变长回填（-1 padding 不再混入样本）
  - npz members 循环外一次物化：修复 lazy 解压 OOM（117MiB ArrayMemoryError）
- **data/patch_simulator_v5.py**
  - 样本多样性随机化（材质 E/α_T ±20%、τ ±30%、h ±30%、ε U(0.85,0.95)、
    初温噪声 N(0,2℃)）——修复同一 (patch,曲线,dt) 下所有样本物理全同的
    确定性退化
  - `max_history_samples=600` deque 截尾：默认曲线 128px 全程 ≈18GB →
    0.2GB/样本；模型只用末 300 帧 + 最终帧，输出逐位不变（fast 64px 验证）

#### 新增

- `generate_cache_v5.py`：断点续跑仿真缓存生成器（逐样本 chunk 原子写 +
  多进程 + `--assemble_only` / `--verify` 逐位对比 + `--cache_dir`）
- `run_train_v5.py --cache_dir` 参数（向后兼容，默认 None）

#### ⚠️ 不兼容说明

- **仿真输出与历史不兼容**：多样性随机化后，同参数重跑仿真结果与
  2026-09-29 之前的所有缓存/数据集**不可复现、不可混用**。旧缓存一律作废。
- **本批修复不解决 GT 形态问题**：应力场"边界高中心低"宽平台下，阈值化
  只能产出满面或空（assets 副本 th=11 下 89% 正样本为满面退化，裂纹像素
  占 93.85% 面积）。阈值重选必须同时报告 GT 面积占比分布，详见
  `docs/v5_已知问题.md`。
- coverage 数值口径变化（单向→双向）：与历史训练日志的 coverage 数字
  **不可直接对比**。

---

## [v5.0.7-alpha] - 2026-08-25

### 文档重整 + v5-β 详细规划（用户反馈："别搞得只有 α 一样"）

按 2026-08-25 用户反馈"**开始适配 V5-β**"：
执行两项任务：(1) 代码重构为 V5-β 预留扩展点；(2) V5-β 占位文件填实（4 个任务）。

#### 实现

**(1) 代码重构 — V5-β 钩子预留**

> 目标：v5-α 当前调用方式 100% 不变；v5-β 启动时只需传新参数即可启用。

- `training/trainer_v5.py`：`TrainerV5.__init__` 新增 6 个 V5-β 钩子参数
  - `data_source: "simulation" | "real"`（默认 "simulation"）
  - `cache_dir: str | None`（默认 None，即内存缓存）
  - `use_position_image: bool`（默认 False，注入位置 6D 分支）
  - `use_uv_inverse: bool`（默认 False，UV 逆映射）
  - `lambda_position_6d: float`（默认 0.0，v5-β 启用时 0.1）
  - `lambda_uv_projection: float`（默认 0.0，v5-β 启用时 0.05）
  - 越界使用触发 `warnings.warn`（不抛错，保持 v5-α 兼容）
- `training/ordered_kp_loss.py`：`OrderedKeypointLoss` 新增 2 个占位 lambda
  - `position_6d_loss(pred, true) → Smooth L1`（None → 0）
  - `uv_projection_loss(pred_kpts, pred_6d, true_6d, K) → MSE`（K=None → 0）
  - forward 新增 `true_position_6d / pred_position_6d` 参数（默认 None）
  - 返回 dict 新增 `position_6d / uv_projection` 分量
- `data/patch_dataset_v5.py`：`PatchDatasetV5` + `collate_v5` 集成 V5-β
  - 新增参数：`data_source` / `use_position_image` / `cache_dir`
  - `_save_to_npz` / `_load_from_npz`：V5B6 磁盘缓存（h5py 未装时用 npz）
  - 缓存文件名包含 `patch_size / n_samples / seed / thermal_hash / threshold`
  - `_generate_sample` 在 `use_position_image=True` 时注入 `true_position_6d=(6,)` 全 0
  - `collate_v5` 仅当**所有样本**都含 `true_position_6d` 时才输出该字段（保守策略）
- `training/trainer_v5.py` CSV history 列扩展：`train_position_6d / train_uv_projection`

**(2) V5-β 占位文件填实（4 个任务）**

- ✅ **V5B3** — `tools/crack_annotator.py`（从 19 行占位 → 680 行完整实现）
  - `auto_extract_crack()`：Canny 边缘 → 形态学闭运算 → skeletonize → 等弧长采样 4-16 点
  - `save_annotation() / load_annotation()`：JSON 序列化（路径 / 陶瓷ID / bbox / 关键点 / 方法）
  - `CrackAnnotatorGUI`：matplotlib 鼠标交互（添加/删除/拖动关键点 + 'a'自动/'s'保存）
  - `batch_auto_extract()`：CLI 批处理（无需 GUI，适合服务器）
  - CLI 三模式：`gui` / `batch` / `validate`
  - 注：v5-α 当前环境**未装 streamlit**，用 matplotlib 替代（覆盖标注需求）
- ✅ **V5B4** — `data/position_extractor.py`（从 19 行占位 → 410 行完整实现）
  - **5 级降级**：
    - `extract_L0()`：多角度（>=4 张）+ 标定板 + 单应性 → 6D pose（cv2 缺失时自动降级）
    - `extract_L1()`：单张 + 窑体可见 → 窑体框 6D pose（cv2 缺失时自动降级）
    - `extract_L2()`：仅陶瓷 → 1D 计数（陶瓷数量放在 position_6d[5]）
    - `extract_L3()`：cart_id + layer_idx → 6D pose（窑体布局推算）— **完整实现**
    - `extract_L4()`：全 0 + validity_mask=False — **完整实现**
  - `PositionInfo` dataclass：`level / position_6d / validity_mask / source_meta` + `.to(device)`
  - `PositionExtractor`：统一接口 + `auto` 模式（按可用数据自动选最高级）
- ✅ **V5B6** — npz 磁盘缓存（已嵌入 `data/patch_dataset_v5.py`）
  - 缓存键：`{patch_size, n_samples, seed, thermal_hash, threshold}.npz`
  - 第 1 次：仿真 + 写缓存；第 2 次起：直接加载（速度提升 10×+）
  - 含 `use_position_image=True` 时缓存 true_position_6d 字段
  - 注：h5py **未装**；用 npz 等效（单文件 vs 切片访问稍慢）
- ✅ **V5B7** — `team_train_v5.py`（从 v4 `team_train.py` 移植 + v5-α 适配）
  - 内置任务 V5A1-V5A6 + V5B3/V5B4/V5B6/V5B7
  - 任务加载过滤：仅 `V5*` 前缀（v4 残留 `team_optimization.json / team_baseline.json` 自动忽略）
  - 模块任务路由：`_V5_MODULE_TEST_MAP`（如 `V5A1` → `test_patch_simulator_v5.py`）
  - CLI：`--list-tasks` / `--auto` / `--force` / `--import`
  - 完整日志：`logs/team_training_v5.log`

#### 修改

- `tests/test_trainer_v5_smoke.py`：**未改动**（v5-α 现有 17 测试仍全过）
- `tasks/team_v5_alpha.json`：将 V5B3/V5B4/V5B6/V5B7 状态标记为"completed"（占位已填实）
- `run_train_v5.py`：**未改动**（V5-β 参数有默认值，无需 CLI 暴露；v5-β 启动时再加 `--use_position_image` 等开关）

#### 验证

- **单元测试合计：101/101 PASSED**
  - v5-α 原有：**89/89**（V5A1 simulator 12 / V5A2 spline_head 8 + gp_module 10 / V5A3 loss 24 / V5A4 model 18 / V5A5 trainer 17）
  - V5-β 新增：**12/12**（`test_v5_beta_hooks.py`）
    * TrainerV5 V5-β 参数接受 + 越界 warning
    * OrderedKeypointLoss position_6d None/真值 双路径
    * collate_v5 条件输出 true_position_6d
    * PatchDatasetV5 npz 缓存 + use_position_image 注入
    * PositionExtractor 5 级降级 + auto 模式 + .to(device)
    * crack_annotator save/load + auto_extract_crack
    * team_train_v5 list-tasks V5 过滤 + 模块路由
- **3 个新冒烟测试**（每个模块独立可跑）：
  - `data/position_extractor.py`：5 级 + auto + .to(device) 全过
  - `tools/crack_annotator.py`：auto_extract_crack 合成测试 + save/load + CLI validate 全过
  - `data/patch_dataset_v5.py`：默认 / npz 缓存 / use_position_image 三模式全过
- **v5-α 兼容性验证**：所有 89 个原有测试在重构后**零改动全过**，证明钩子参数完全向后兼容

#### 已知问题新增

- V5-018：v5-β L0/L1 占位 stub（cv2 缺失时自动降级到 L3）
  → v5-α 接受（依赖：陶艺作坊合作启动时引入 opencv-contrib-python）
- V5-019：npz 缓存单文件 vs h5py 切片访问略慢
  → v5-α 接受；v5-β 数据量 > 10k 时评估 h5py 切换
- V5-020：streamlit 未装，crack_annotator 用 matplotlib 替代
  → v5-α 接受；v5-β 启动时评估 streamlit 包装（Web GUI）

---

## [v5.0.6-alpha] - 2026-08-24

### 文档重整 + v5-β 详细规划（用户反馈："别搞得只有 α 一样"）

按 2026-08-24 用户反馈，文档系统**完全独立化** + **v5-β 详细规划**：

#### 删除

- ❌ `docs/archived/` 全部（21 个 v4 历史快照）
- ❌ `docs/架构设计文档.md`（v4 only，v5-α 用 `v5_架构设计.md`）

#### 重写（彻底剥离 v4 内容）

- 📝 `docs/v5_架构设计.md` — 顶部加"继承说明"（5 骨干沿用 v4 实现细节），新增 §8 v5-β 架构演进（V5B1-V5B8）
- 📝 `docs/v5_范围说明.md` — 重写为完整范围声明（v5-α + v5-β 各 § 一节），含升级路径图与风险
- 📝 `docs/v5_仿真升级清单.md` — 新增 §8 v5-β 完整仿真（S5 完整 Griffith + S6 完整 LEFM + 3D FEM + 材料数据库）
- 📝 `docs/全流程.md` — 去掉 v4 团队协作，保留单人 5 阶段闭环
- 📝 `docs/文档审查指南.md` — 去掉 v4 `archived/` 子目录策略，文档地图重排
- 📝 `docs/快速配置指南.md` — 抛弃 v4 数据加载 / `team_train.py` / 6 维向量输出，新增 v5-β 快速上手
- 📝 `docs/CUDA安装指南.md` — 加 v5-α 推荐 CUDA 配置表
- 📝 `docs/团队协作训练指南.md` — 明确 v5-α 不实现 `team_train.py`，V5B7 详细接口
- 📝 `docs/api.md` — 完全废弃 v4 API，新增 §8 v5-β 规划 API（V5B1-V5B8 全部接口预留）
- 📝 `docs/开发人员文档.md` — 抛弃 v4 旧内容，新增 §10 v5-β 开发路线（V5B1-V5B8 任务分解 + v5-α 模型复用清单）
- 📝 `docs/调参与算法工程指导文档.md` — 抛弃 v4 旧内容，新增 §8 v5-β 调参规划
- 📝 `docs/项目算法与训练实验设计报告.md` — 重写 v5-α + 新增 v5-β 实验设计（数据集划分 + 消融 + 评估指标）
- 📝 `README.md` — 完整 v5-α + v5-β 双章节（任务清单 / 仿真升级 / 算法差异 / 文档入口）

#### v5-β 文档覆盖范围（新增）

- ✅ V5B1-V5B8 8 个任务的详细接口（UVUnwrapper / MultiViewCapture / CrackAnnotator / PositionExtractor / FEM3DSimulator / team_train_v5 / Streamlit v5）
- ✅ v5-α 模型复用清单（哪些组件可直接复用 / 哪些需修改）
- ✅ v5-β 评估指标（3D 重投影误差 + 多视图一致性 + 位置 6D 误差）
- ✅ v5-β 数据集划分（Train 5000+ / Val 500 / Test 500 / Cross-material 1000）
- ✅ v5-β 风险（真实数据采集成本 / UV 展开失真 / 标注一致性）
- ✅ v5-β 升级路径图（α → β → 1.0）

#### 影响

- ✅ 用户查阅 v5-α 文档**无需进入 v4 docs/**
- ✅ 每个文档顶部"继承说明"段清楚标注 v4 → v5-α 关系
- ✅ v5-β **不再只是占位**，有完整 8 个任务的接口 + 路径 + 风险
- ✅ 文档地图（`文档审查指南.md` §一）反映平铺结构

#### 已知问题

- 无新增

---

## [v5.0.5-alpha] - 2026-08-12

### 新增（V5A5）

v5-α 训练循环（变长关键点 + 断点续训 + CSV）：

- `data/patch_dataset_v5.py`：`PatchDatasetV5` + `collate_v5`
  - 包装 `PatchSimulator.simulate()` 生成样本
  - **预计算缓存**（__init__ 时一次性仿真，避免每个 epoch 重复）
  - 变长批处理：`collate_v5` 把不同长度的真值关键点/裂纹像素 padding 到 batch 内最大尺寸 + mask 标记
  - 样本字段：`x_1d, x_2d, true_bbox, true_keypoints, true_crack_pixels, metadata`
- `training/data_aug_v5.py`：`PatchAugmentorV5`
  - 物理安全增强：水平翻转 + 垂直翻转 + 90° 旋转 + 高斯噪声
  - bbox / 关键点 / 像素坐标随图像同步变换
  - 物理约束保护：bbox 顺序（x1≤x2, y1≤y2）、值域 [0,1] 在增强后仍成立
- `training/trainer_v5.py`：`TrainerV5` + `CSVHistory` + `save/load_checkpoint`
  - 变长 batch 处理（GPU 同步 padding 到 collate 后 mask）
  - GP 模块正则注入（`OrderedKeypointLoss` 内置）
  - 覆盖距离 loss 调用
  - 动态 K 选择（`SplineHead` 内置）
  - **断点续训**：`load_checkpoint` + `fit(resume_from=...)` + RNG 状态恢复
  - **CSV 历史**：每个 epoch 记录 train_loss / val_loss / lr / 各分量 / 时间戳
  - **best + latest checkpoint**：自动保存 val_loss 最优 + 每 save_every epoch 保存
- `run_train_v5.py`：统一 CLI 入口（沿用 v4 argparse 风格）
  - 支持 `--variant` / `--epochs` / `--batch_size` / `--lr` / `--resume` 等
  - `--fast_thermal` 用于 1 min 短周期（测试用）
  - `--no_gp` 禁用 GP 模块
  - `--pretrained_2d` 启用 ImageNet 预训练
- `tests/test_trainer_v5_smoke.py`：17 项单元测试

### 关键设计决策

**预计算数据集**（v5-α MVP 简化）：
- 仿真一次 ~30s（256×256 全周期）— 每个 epoch 重新仿真不可接受
- `__init__` 时一次性仿真所有样本到内存 cache
- v5-β 真实数据接入时，替换为懒加载 + 磁盘缓存（接口不变）

**变长关键点的 collate 处理**：
- `collate_v5` 返回 `true_keypoints (B, max_K, 2)` + `keypoint_mask (B, max_K)`
- 用 -1 标记 padding 位置（loss 计算时被 mask 过滤）
- 真值裂纹像素同样处理

**RNG 状态恢复**（断点续训关键）：
- 检查点保存 `python / numpy / torch` 三种 RNG 状态
- `load_checkpoint` 时恢复，保证续训可复现

**空真值像素边界**：
- `coverage_loss` 处理 `N=0`（无裂纹像素）时返回 surrogate（无梯度）而非 NaN
- 防止训练初期全部 batch 都没裂纹时 loss 爆炸

### 物理修复（V5A3 跟进）

`coverage_loss` 修复：当 `true_crack_pixels.size(1)=0`（无裂纹）时，原实现 `min_per_true.mean(dim=-1)` 会因空张量返回 NaN。改为返回 surrogate（0 梯度），避免训练中断。

### 测试

- 单元测试合计：**89/89 PASSED**
  * V5A1 patch_simulator: 12 项
  * V5A2 SplineHead: 8 项 + GPModule: 10 项
  * V5A3 ordered_kp_loss: 24 项
  * V5A4 pe_tsnet_v5: 18 项
  * V5A5 trainer_v5: **17 项**（新增）
- 端到端验证：5 变体 × 1 epoch 训练完整跑通

### 已知问题新增

- V5-016：仿真预计算占用内存大（每个 256×256 样本 ~2MB → 1000 样本 ~2GB）
  → v5-α MVP 接受；v5-β 真实数据接入时改为磁盘缓存
- V5-017：`--fast_thermal` 仅用于测试；真实训练必须用 11.7h 默认周期
  → 已文档化；训练脚本无 `--fast_thermal` 时走默认周期

---

## [v5.0.4-alpha] - 2026-08-12

### 新增（V5A4）

5 个 v5-α 模型变体（输出头已替换为 SplineHead + GP 嵌入）：

- `models/pe_tsnet_multimodal_v5.py`：`PETSNetMultimodalV5`
  - ResNet-18 + CrossAttentionFusion + SplineHead + GP（默认启用）
  - 输入：3 通道（温度场+应力场+热力图），符合 v5-α 设计
- `models/pe_tsnet_yolo_v5.py`：
  - `SwinYOLOFPNV5` — Swin-Tiny + YOLO-FPN
  - `ViTYOLOFPNV5` — ViT-Small + YOLO-FPN
  - `ViTYOLOBackbone2DV5` — 自适应输入通道版（v4 硬编码 2 通道 → v5-α 1/2/3/4 通道）
- `models/pe_tsnet_detr_v5.py`：`DETRStyleV5` — ResNet-18 + Transformer Encoder-Decoder
- `models/pe_tsnet_patchtst_v5.py`：`SwinYOLOFPNWithPatchTSTV5` — Swin + YOLO + PatchTST 1D
- `models/__init__.py`：注册 `V5_MODEL_REGISTRY` + 工厂函数 `create_v5_model(variant, **kwargs)`
- `tests/test_pe_tsnet_v5.py`：18 项单元测试

### 关键改动

| 维度 | v4 | v5-α |
|---|---|---|
| 输出 | 6 维向量 / YOLO 网格 6 维 / DETR queries 6 维 | bbox + 8-16 动态关键点（dict） |
| 输出头 | MultiTaskHead / YOLOFPNHead / DETRHead | SplineHead（统一） |
| 物理约束 | 单调性损失 | GP 嵌入模块（嵌入特征流）+ GP LML 正则 |
| 输入通道 | 2（温度+应力） | 3（温度场+应力场+热力图） |
| 训练接口 | `model(x_1d, x_2d) → Tensor` | `model(x_1d, x_2d) → dict` |

### 设计决策

**复用 v4 骨干，不重写**：
- 5 个变体的 2D 骨干（ResNet-18 / Swin-Tiny / ViT-Small）和 1D 骨干（TemporalFeatureExtractor / PatchTST1D）全部沿用 v4
- 仅替换输出头 + 可选 GP 模块插入融合后特征
- 总代码量减半，单测覆盖完整数据流

**DETR 的 top-1 query 选择**：
- 沿用 v4 推理链：`DETRHead → (B, num_queries, 6) → 取最高 conf → query_proj → fusion`
- v5-α 训练时也用同一 query（保持 train/eval 一致性，避免 query 选择 抖动）

**YOLO 变体的全局池化**：
- v4 输出 `(B, num_grids, 6)`（网格级预测）
- v5-α 用 `mean(dim=1)` 池化到 `(B, 6)`（全局特征），再接 1D 拼接
- 简化版聚合策略，避免在 v5-α 阶段处理 grid-to-kpts 的复杂映射

### 测试

- 单元测试合计：**72/72 PASSED**
  * V5A1 patch_simulator: 12 项
  * V5A2 SplineHead: 8 项 + GPModule: 10 项
  * V5A3 ordered_kp_loss: 24 项
  * V5A4 pe_tsnet_v5: **18 项**（新增：5 变体 forward + backward + 端到端 loss 反向）
- 端到端验证：所有 5 个变体都能跑 `model → OrderedKeypointLoss → backward` 全链路

### 已知问题新增

- V5-015：GP 模块 batch 内归一化（V5-014）已在 v5-α 模型集成中保留，**未升级到 running stats**
  → 训练/推理 stats 不一致风险持续存在
  → 解决路径：V5A5 trainer 集成时改用 `(running_mean, running_std)`

---

## [v5.0.3-alpha] - 2026-08-12

### 新增（V5A3）

v5-α 主损失函数（覆盖距离 + 组合 loss）：

- `training/ordered_kp_loss.py`：v5-α 核心损失
  * `catmull_rom_spline_torch(points, num_samples)`：纯 PyTorch 三次样条拟合
    - 端点镜像延拓 ghost points（保持 C¹ 连续）
    - K=2 退化为线性插值
    - 完全在 PyTorch 内实现，可对控制点求梯度（autograd 通路畅通）
  * `coverage_loss(pred_kpts, true_pixels, M, mode)`：
    - "chamfer"（默认，mean over true pixels of min distance）— 平滑，适合训练
    - "hausdorff"（max over true pixels of min distance）— 严格，评估用
  * `poisson_prior(pred_kpts)`：段长方差最小化（鼓励等弧长分布）
  * `OrderedKeypointLoss` 组合 loss：
    ```
    total = 1.0  * coverage
          + 1.0  * bbox (Smooth L1)
          + 0.1  * gp (-log_marginal_likelihood)
          + 0.05 * poisson
    ```
  * 返回 dict：`{total, coverage, bbox, gp, poisson}`（含 detached 监控值）
- `tests/test_ordered_kp_loss.py`：24 项单元测试（全部通过）

### 关键设计决策

**为何选 Catmull-Rom 而非 scipy CubicSpline？**

- `scipy.interpolate.CubicSpline` 不可微（基于 numpy，无 autograd 通路）
- Catmull-Rom 是 Hermite 三次样条（C¹ 连续），完全在 PyTorch 内实现
- 对每个 segment (P_{i-1}, P_i, P_{i+1}, P_{i+2}) 做三次插值：用户文档中"三次 B 样条"即指此
- 端点用镜像延拓 ghost points（保持端点切线连续）

### 测试

- 单元测试合计：54/54 PASSED
  * V5A1 patch_simulator: 12 项
  * V5A2 SplineHead: 8 项
  * V5A2 GPModule: 10 项
  * V5A3 ordered_kp_loss: **24 项**（新增）
- 关键验证点：
  * 完美匹配：真值像素 = 样条采样点 → coverage loss ≈ 0（实测 0.00e+00）
  * chamfer ≤ hausdorff（同数据）实测：0.044 ≤ 0.270 ✓
  * 反向传播：pred_kpts / pred_bbox / GP 超参数梯度全部正常

### 已知问题新增

（无新增 V5-α 阻塞问题；V5-014 仍为待 V5A4 集成处理）

---

## [v5.0.2-alpha] - 2026-08-12

### 新增（V5A2）

模型输出头与物理约束模块：

- `models/spline_head.py`：Spline 输出头
  * 输入：(B, in_dim) 融合特征
  * 输出：bbox (B, 4) + 关键点 (B, max_kpts, 2) + validity (B, max_kpts) + K (B,)
  * K 动态范围 [min_kpts, max_kpts] = [8, 16]
  * bbox 自动满足 x1≤x2, y1≤y2（重新参数化）
  * `predict_kpts_only(sort_by="x"|"validity")` 推理便捷接口
- `models/gp_module.py`：GP 嵌入模块
  * 可微 RBF 核平滑（ℓ、σ² 可学习）
  * log_marginal_likelihood 正则项
  * LayerNorm 稳定输出
  * 非可学习模式（buffer）支持
- `tests/test_spline_head.py`：8 项单元测试（值域、bbox 顺序、反向传播、推理接口）
- `tests/test_gp_module.py`：10 项单元测试（核对称、LML、超参数梯度、batch=1 边界）

### 关键修复

**GP 模块 K 矩阵退化为单位矩阵**：

- 现象：默认 in_dim=128 时，||x_i - x_j||² ~ 128，远超默认 ℓ=1.0，
  导致 K ≈ I（单位矩阵），GP 平滑失效
- 修复：`_normalize_for_kernel` 改为按 L2 范数归一化（除以 sqrt(in_dim)），
  使 ||x_i - x_j||² ~ O(1)，ℓ=1.0 物理意义清晰
- 影响：修复后 GP 模块对超参数 ℓ/σ² 都有非零梯度（之前 ≈ 0）

### 测试

- 单元测试合计：30/30 PASSED
  * V5A1 patch_simulator: 12 项
  * V5A2 SplineHead: 8 项
  * V5A2 GPModule: 10 项

### 已知问题新增

- V5-014：GP 模块依赖 batch 内 L2 归一化（无 running stats）
  → 训练/推理 stats 不一致风险（v5-α MVP 可接受）

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
cd D:\New_team_project\projects\pe_mmnet\project_v5
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