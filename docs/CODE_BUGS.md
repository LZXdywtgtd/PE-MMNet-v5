# 已知 Bug 清单（CODE_BUGS）

> PE-MMNet v4 已知但**当前不修**的 code bug。每条都注明位置 + 状态 + 不修原因。
>
> 本文件不参与推送；只在 `docs/流程.md` 的循环中 review。

---

## Schema

每条 bug 字段：

- **ID**：B001, B002, ...（按发现顺序）
- **状态**：`已知` / `已记录但未修复` / `已修复`（修复时填 commit hash）
- **代码位置**：`file:line` 或函数名
- **现象**：什么场景下出错
- **影响**：功能/性能/数据丢失
- **不修原因**：本次故意不修

---

## B001 — `--batches` argparse 接受但代码不使用

- **状态**：已知
- **位置**：`run_train.py` argparse（约 :2404）
- **现象**：`--batches foo bar` 解析为列表后从未被任何代码读取
- **影响**：用户传这个 flag 无任何效果，但 CLI 仍然接受，不会报错
- **不修原因**：保持向后兼容；旧脚本可能还在传这个 flag

## B002 — `--freeze_backbone` 和 `--freeze_2d` 无法关闭

- **状态**：已知
- **位置**：`run_train.py` argparse（约 :2480-2490）
- **现象**：用 `store_true` + `default=True` → CLI 只能让它更 True（已经是 True），无法传 `--no-freeze-backbone` 关闭
- **影响**：骨干始终冻结，无法微调
- **不修原因**：当前架构下冻结骨干是默认且正确的策略；不暴露禁用选项避免误用

## B003 — `--lambda_mono` 对 4 个新变体静默忽略

- **状态**：已知
- **位置**：`run_train.py:1462-1490` 的 loss 选择逻辑
- **现象**：`swin_yolo` / `vit_yolo` / `detr` / `swin_yolo_patchtst` 4 个变体构造 `YOLOLoss` / `DETRLoss` 时**硬编码** `lambda_mono=0.1`，不读 `args.lambda_mono`
- **影响**：用户传 `--lambda_mono 0.3` 对新变体无效
- **不修原因**：硬编码值经过调参验证是合理的；统一通过改 `mono_loss.py` 中的默认值更安全

## B004 — `tasks/*.json` 顶层 dict 形式静默丢弃

- **状态**：已知
- **位置**：`team_train.py:483` 的 `load_tasks_from_files`
- **现象**：只接受顶层数组。`{"tasks": [...]}` 包装形式会被静默丢弃，菜单中看不到任何任务
- **影响**：用户按"标准 JSON 包装"格式写任务清单时无错误提示，导致 0 任务运行
- **不修原因**：当前格式已经稳定；加 schema 校验属于 feature 改动，本轮不修

## B005 — `team_baseline.json` 临时为 `--epochs 2`

- **状态**：临时态（用户约定）
- **位置**：`tasks/team_baseline.json` / `tasks/team_optimization.json`
- **现象**：所有任务 `--epochs 2`（快速验证用），但 `time` 字段仍写 "~12-24 小时"
- **影响**：跑生产会"看起来"12-24 小时，实际 5-10 分钟
- **不修原因**：用户在用 2 epoch 验证流程；跑生产前手动改回 150

## B006 — `DEFAULT_TRAIN_TASKS` 内置任务缺 `subdir`

- **状态**：已知
- **位置**：`team_train.py:180` 的 `DEFAULT_TRAIN_TASKS`
- **现象**：内置 13 个任务定义没 `subdir` 字段，外部 JSON 文件有。`merge_tasks` 合并时以外部为准，所以实际影响小
- **影响**：理论上前置检查写路径与 JSON 任务不一致
- **不修原因**：外部 JSON 始终胜出，内置任务被 JSON 覆盖后无问题

## B007 — `L2+` 等级 rank 等于 `L2`

- **状态**：已知
- **位置**：`team_train.py:476` 的 rank map
- **现象**：`{L1:1, L1+:2, L2:3, L2+:3, L3:4}` → L2+ 与 L2 同级（rank 3）
- **影响**：L2+ 任务（如分阶段+坐标注意力，需要更多显存）在 L2 机器上**永不警告**，但可能实际 OOM
- **不修原因**：rank map 是 v4.6.10 之前的妥协；L2+ 实际显存需求接近 L2；改为不同 rank 需要重新校准所有任务的 `level` 字段

## B008 — `get_completed_tasks` 只扫 `*_best.pt`

- **状态**：已知
- **位置**：`team_train.py:383` 的 `get_completed_tasks`
- **现象**：只检查 `*_best.pt` 的 `is_complete` 字段。如果某个任务的 `is_complete` 只在 `_last.pt` 上（如老格式迁移后 best 文件丢失），会被认为未完成
- **影响**：罕见的 false-negative；可通过 `_get_task_checkpoint_info` 缓解
- **不修原因**：当前 best.pt 始终存在；`is_complete` 应同时写在 best 和 last

## B009 — `import_checkpoint` 复制到 `checkpoints/` 根目录

- **状态**：已知
- **位置**：`team_train.py:935` 的 `import_checkpoint`
- **现象**：队友的 `.pt` 文件复制到 `checkpoints/` 根目录而非 `checkpoints/{variant}/{subdir}/`
- **影响**：导入后 `get_completed_tasks` 找不到（它只扫 `*_best.pt`，不论目录），需要手动 mv 到正确位置
- **不修原因**：导入是手动操作，用户可手动调整；自动化路径分类属 feature 改动

## B010 — 7 天清理会删除 `team_training.log`

- **状态**：已知
- **位置**：`run_train.py:2516-2521` 的日志清理
- **现象**：启动时清理 `logs/` 中 7 天前的所有 `.log` 文件，**不分内容**——`team_training.log` 也会被删
- **影响**：团队训练的历史日志被清
- **不修原因**：保留历史日志需要专门的归档机制，本轮不引入

## B011 — `launcher.py` 和 `train_launcher.py` 功能重叠

- **状态**：已知
- **位置**：项目根
- **现象**：`launcher.py`（889 行）和 `train_launcher.py`（286 行）都是"交互式启动器"，功能大量重叠
- **影响**：新用户不知道该用哪个
- **不修原因**：删除 `launcher.py` 是 feature 改动；本轮文档说明"以 train_launcher.py 为准"

## B012 — `SwinYOLOFPNWithPatchTST` 和 `PatchTST1D` 未在 `models/__init__.py` 导出

- **状态**：已知
- **位置**：`models/__init__.py`
- **现象**：`from models import SwinYOLOFPNWithPatchTST` 会 ImportError
- **影响**：只能通过 `run_train.py --variant swin_yolo_patchtst` 间接使用
- **不修原因**：直接导入会绕过注册流程；保持单一入口

## B013 — `run_train.py` 中部分 CLI flag 没有 help 文本

- **状态**：已知
- **位置**：`run_train.py` argparse（约 :2404-2507）
- **现象**：`--dropout`, `--triple_channel` 等少数 flag 缺 `help=""` 参数
- **影响**：`--help` 输出少几行
- **不修原因**：非关键；用户主要用快速配置指南和团队指南查参数

---

## 已修复 bug 历史

（暂无）
