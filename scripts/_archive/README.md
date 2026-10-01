# scripts/_archive/ — 历史一次性脚本（不保证可直接复跑）

v5-α 十一轮诊断/实验/可视化的全量历史件，按归档时**原样保留**
（含旧相对路径与 cwd 依赖，如 `_diag_pca.py` 无 PROJECT_ROOT 自定位）。

**这些脚本的价值是"当时发现过什么"**——全部结论已落
`CHANGELOG.md` 与 `docs/v5_已知问题.md` / `docs/v5b_立项依据.md`；
复现某个结果请走 `git log` 历史版本，不要指望归档件开箱即跑。

## 2026-10-01 重组注意：`_exp_*` 场缓存已搬家

实验场缓存目录移至 `output/exp_fields/`：
`_exp_sigma_fields` → `output/exp_fields/_exp_sigma_fields`，
`_exp_default_curve_fields{,_v2}` 同理。

以下 **6 个脚本硬编码旧路径，复跑前需自行改路径**（引用
`_exp_sigma_fields` / `_exp_default_curve_fields_v2`）：

- `_exp_default_curve_lowth.py`
- `_exp_default_curve_spot.py`
- `_exp_default_curve_spot_v2.py`
- `_exp_field_cache_sweep.py`
- `_exp_rescan_post_v524.py`
- `_exp_v5025_cause.py`

（生产侧唯一引用方 `data/_dprime_generator.py` 自检②已同步改新路径，
不在本目录。）

## 内容清单

| 类别 | 件数 | 说明 |
|---|---|---|
| `_diag_*` | 15 | 九轮诊断（variance79 后继/选项 C 五步/位移方向等；活跃后继在 `scripts/` 与 `scripts/lib/`） |
| `_exp_*` | 18 | 仿真路线实验（周期边界/边界调制/D' 统计/th 扫描等）+ results.jsonl×2 |
| `_viz_*` | 4 | 一次性可视化（A1 对比图/tail7pct 等；共享件 `_viz_common` 在 `scripts/lib/`） |
| bench/smoke | 5 | `_bench_sim/_bench_threshold/_smoke_orderv` 等 |
| 过夜日志 | 18 | `_gen_*`/`_train_*`/`_smoke_v5a6.log`（gitignored 的纯 mv，不留 git 历史） |
| 杂件 | `_eval_criteria34.py`（被 `_eval_s3_gtfix` 系列取代）、`_gt_vis.txt`、`_diag_variance79_vis.png` |
