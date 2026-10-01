# legacy_v4/ — v4 时代入口（历史参照，勿用于 v5）

v4 → v5 fork 时留下的入口文件。**v5 一律不用**，保留仅作历史参照
（与 v4 家族仓库逐字节同源的有意 fork，改 bug 时两边需同步）。

| 文件 | 说明 |
|---|---|
| run_train.py | v4 单变体训练入口（122KB，被 tools/batch_train_gui 懒导入） |
| launcher.py | v4 图形启动器（导入 legacy_v4/tools/batch_train_gui） |
| team_train.py | v4 团队多机训练入口 |
| train_launcher.py | v4 CLI 启动器 |
| verify_checkpoints.py | v4 checkpoint 校验器 |
| run.bat / run.ps1 | v4 启动脚本（$PSScriptRoot 相对路径，可直接运行） |
| tools/batch_train_gui.py | v4 批量训练 GUI（与上者同目录自洽） |

v5 对应入口在项目根：`run_train_v5.py`、`team_train_v5.py`、
`generate_cache_v5.py`。
