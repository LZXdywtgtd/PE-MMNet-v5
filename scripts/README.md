# scripts/ — 活跃脚本区

按"活不活"分类的三个子区（2026-10-01 重组）：

```
scripts/
├── *.py          # 活跃入口（可直接跑）：收官评估 _eval_s*、GT 验收 _spotcheck、活跃诊断 _diag_*、汇报演示 _v5a6_demo
├── lib/          # 共享库（被导入，不直接跑）：_diag_variance79 / _diag_v5b_direction2 / _viz_common
├── _archive/     # 历史一次性（不保证复跑，见其 README）
└── README.md
```

## 运行方式

从项目根运行（PROJECT_ROOT 已按两级 dirname 自定位，与 cwd 无关）：

```bash
python scripts/_eval_s3_gtfix.py          # 收官基线判据 3/4 评估
python scripts/_spotcheck_gt_v032.py      # GT 全量 L 形扫描验收
python scripts/_v5a6_demo.py              # 物理因果链汇报演示
```

## 导入约定

- 入口脚本同时注入两个 sys.path：`PROJECT_ROOT`（项目根，导 models/training/data）与
  `PROJECT_ROOT/scripts/lib`（导共享库裸名 `_diag_variance79` 等）。
- `lib/` 内部互导保持裸名（`from _diag_variance79 import …`），依赖 lib 在 sys.path 上。
- 若写新脚本既导 lib 又导 `scripts/` 平层的其他活跃脚本，需第三个 sys.path 条目
  （`PROJECT_ROOT/scripts`）——目前无此用例。

## 待转正

`_diag_aug_marker.py`（增广标记点自洽性测试，rot90 修复 V5-031 的守护）：
立项文档 §四建议转正式单测——转正时改名 `tests/test_aug_marker.py` 一并动作。
它现在按**活跃诊断脚本**对待（3/3 PASS 可随时跑），不是归档件。
