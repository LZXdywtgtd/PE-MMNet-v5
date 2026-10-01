# PROJECT_INDEX —— project_v5（v5-α 收官 · v5-β 待启动）

> **路径**：`D:/New_team_project/projects/pe_mmnet/project_v5/`
> **定位**：PE-MMNet 第五代（独立 git，remote PE-MMNet-v5.git / branch main）。v4 fork + Physics-Informed 强约束转向：GP/泊松过程先验、Catmull-Rom 样条+Hausdorff 覆盖距离损失、8-16 动态关键点+bbox 变长有序轨迹输出；256×256 patch 仿真器自供训练数据。
> **当前状态**：**v5-α 正式收官**（干净 GT 口径：基线 197/197=100%，5 seed 全 100%，判据 4 PASS，V5-030~V5-033 全闭环）；v5-β 唯一主线 = 真实数据泛化（`docs/v5b_立项依据.md` v3.2.1）。
> **索引版本**：v3 · 2026-10-01 · 结构重组后重写（v2 2026-09-13 已过时：其"128 文件 1.72GB"与删除候选清单均已被本轮执行消化）

---

## 阅读顺序

- **了解 v5 设计** → README.md → docs/v5_架构设计.md → docs/v5_范围说明.md
- **了解 v5-α 为什么收官、v5-β 为什么立项** → docs/v5b_立项依据.md（唯一入口）
- **看 v5-α 十一轮诊断史** → docs/v5_已知问题.md（V5-001~V5-034）+ CHANGELOG.md
- **看物理仿真自供数据** → data/patch_simulator_v5.py（S1-S7）+ data/_dprime_generator.py（D' 边界调制）
- **复现收官数字** → `python run_train_v5.py --variant resnet18 --epochs 150 --dprime --cache_dir logs/sim_cache --fast_thermal`（口径见立项文档 §八）→ 评估走 `scripts/_eval_s3_gtfix.py` 等
- **物理因果链展示（汇报）** → `scripts/_v5a6_demo.py` → logs/training_history/v5a6_demo/

对账（2026-10-01，git ls-files）：根 8 + .github 2 + data 13 + docs 15 + legacy_v4 8 + models 13 + scripts 53 + tasks 4 + tests 11 + tools 8 + training 6 + utils 3 + benchmark_results 1 = **145 tracked**；磁盘大头 logs/ 7.8GB（training_history 收官产物 ~5.9GB + sim_cache 869MB）+ _backups/ 1.1GB（用户备份，未跟踪）。

---

## 第 0 层：根目录（8 文件）

| 文件 | 是干啥的 | 建议 |
|---|---|---|
| `README.md` | v5 门面（2026-10-01 更新：新目录结构 + 收官版本号） | 保留 |
| `PROJECT_INDEX.md` | 本文件 | 保留 |
| `CHANGELOG.md` | v5 演进史（V5-001~V5-033 + 结构重组 v5.0.17） | 保留 |
| `config.json` / `example_config.json` | 运行配置 | 保留 |
| `requirements.txt` | 依赖 | 保留 |
| `run_train_v5.py` | **v5-α 统一训练脚本**：`--variant` + `--dprime`（V5-033 P0 缓存键）+ `--train_seed`（数据/训练种子分离）+ `--resume`；启动横幅打印缓存文件与命中状态 | **保留（v5 核心）** |
| `team_train_v5.py` | **团队协作训练（V5B7）**：任务路由 tasks/team_v5_alpha.json；`_V5_MODULE_TEST_MAP` 指 tests/ 现存路径 | **保留（核心）** |
| `generate_cache_v5.py` | **缓存生成器**：断点续跑（chunks/）+ `--dprime` 懒导入 `data._dprime_generator` | **保留（核心）** |

## 第 1 层：data/（13 文件）—— 数据层（物理仿真自供数据）

| 文件 | 是干啥的 |
|---|---|
| `patch_simulator_v5.py` | **patch 仿真器（V5A1，S1-S7）**：FDM 温度场+Stefan-Boltzmann 边界+Maxwell 粘弹性应力+裂纹萌生；V5-032 剥框修复（`_strip_boundary_frame` 单边判据）在此 |
| `patch_dataset_v5.py` | **数据集（V5A5）**：x_1d 时序+x_2d 三通道+x_key 热力图+kpts/bbox 标签；V5-033 P0 缓存键（`_cache_filename` 含 dp/sd+gs/og+h{th} 变体标记；dprime miss→FileNotFoundError 双拒绝守卫）在此 |
| `_dprime_generator.py` | **D' 边界调制生成器（产线依赖）**：DPrimePatchSimulator 子类化仿真器，`generate_cache_v5.py --dprime` 懒导入；`--selftest` 位级安全网（σ 场 vs `output/exp_fields/_exp_sigma_fields` 逐位） |
| `position_extractor.py` | 位置图像 5 级降级管线（V5B4，v5-β 钩子） |
| `surface_radiation.py` / `thermal_profile.py` / `viscoelastic.py` | 辐射（S2）/三段曲线（S4）/Maxwell 应力（S3）物理件 |
| `dataset_multimodal.py` / `batch_validator.py` / `data_diagnosis.py` / `image_preprocessor.py` | v4 同源 fork 件 |

## 第 2 层：models/（13）+ training/（6）+ utils/（3）

- **models/** v5 核心 6：`spline_head.py`（V5A2 样条头）、`gp_module.py`（GP 嵌入）、`pe_tsnet_multimodal_v5.py`（resnet18，**收官变体**）、`pe_tsnet_detr_v5.py`、`pe_tsnet_yolo_v5.py`（swin/vit）、`pe_tsnet_patchtst_v5.py`；v4 同源 fork 7。
- **training/** v5 核心 3：`trainer_v5.py`（变长 batch 训练循环，`--train_seed` 在此）、`ordered_kp_loss.py`（覆盖距离损失三修：负样本掩蔽/双向 chamfer/kpt_direct）、`data_aug_v5.py`（V5-031 rot90 标签同步修复本体）；v4 同源 fork 3。
- **utils/** 3：v4 同源 config/console。

## 第 3 层：scripts/（53 tracked）—— 2026-10-01 重组新家

| 子区 | 内容 | 性质 |
|---|---|---|
| `scripts/` 平层（13 入口） | `_eval_s3_gtfix/_eval_s3_dp/_eval_s4_seeds/_eval_s5_variants`（收官评估）、`_spotcheck_gt_v032`（GT 验收）、`_diag_augfix2_sanity/_diag_augfix_eval/_diag_cov_discrepancy/_diag_displacement_mode/_diag_stock6x/_diag_trainer_val`（活跃诊断）、`_diag_aug_marker`（增广自洽，立项文档 §四建议转正式单测）、`_v5a6_demo`（汇报演示） | **可直接跑**：PROJECT_ROOT 已改两级 dirname；导 lib 的 9 个已加 `sys.path.insert(..., "scripts/lib")` |
| `scripts/lib/`（3 库） | `_diag_variance79`（CKPT_BASE/VAL_CACHE/PX/geom_features…）、`_diag_v5b_direction2`（cham_of/UPPER/gt_cells_of/REQ…）、`_viz_common`（字体/加载/推理公共件） | **被导入不直接跑**；库间裸名互导原样保留 |
| `scripts/_archive/`（62 件） | v5-α 十一轮诊断/实验/可视化全量历史：`_diag_*` 15、`_exp_*` 18、`_viz_*` 4、bench/smoke 5、过夜 .log 18、results.jsonl 2、`_eval_criteria34`、_gt_vis.txt、_diag_variance79_vis.png | **一次性历史件**：按归档原样保留（含旧相对路径，不保证直接复跑）；结论已落 CHANGELOG/演进史，复现走 git 历史 |

## 第 4 层：legacy_v4/（8）—— v4 入口历史参照

`run_train.py`（122KB）、`launcher.py`、`team_train.py`、`train_launcher.py`、`verify_checkpoints.py`、`run.bat`、`run.ps1` + `tools/batch_train_gui.py`（与 launcher 同目录自洽）。**v5 一律不用**，详见 legacy_v4/README.md。`tools/crack_annotator.py`（v5-β 关键件，team_train_v5 + tests 引用）**留在 tools/**。

## 第 5 层：tests/（11）

| 文件 | 是干啥的 |
|---|---|
| `test_*.py` ×8 | pytest 单测 101 项（仿真器/损失/GP/样条头/模型/训练器冒烟/β 钩子） |
| `_test_v032_frame_strip.py` | V5-032 剥框手工验证 9/9（`python tests/_test_v032_frame_strip.py` 指名运行；T6 经 `data._dprime_generator` 端到端） |
| `_test_v033_cache_key.py` | V5-033 P0 缓存键验证 5/5（新键命中/miss 报错/双拒绝/gs-og 独立） |

> 下划线开头 pytest 不收集——手工验证件与单测的地位区别刻意保留在命名上。

## 第 6 层：tools/（8）、tasks/（4）、docs/（15）、其他

- **tools/**：`crack_annotator.py`（V5B3 在用）+ `patch_simulator_visualizer.py` + v4 同源工具 6（streamlit_app/visualization/structure_analyzer/cleanup_old_ckpts/fix_save_reason/test_contour_remover）。
- **tasks/**：`team_v5_alpha.json`（V5A1-A6）+ v4 同源配方 3。
- **docs/**（15 tracked）：v5 专属 7（架构设计/范围说明/仿真升级清单/已知问题/**v5b_立项依据**/演进史/api.md）+ v4 继承 8；另有 `0902临时汇报用入口/`（未跟踪）。
- **.github/workflows/feishu-notify.yml**：push（master+main）+ release 飞书通知。
- **benchmark_results/ checkpoints/**：仅 .gitkeep（产物实际写 logs/training_history/，结构占位）。
- **output/**：v4 fork 图 + `exp_fields/`（2026-10-01 新家：`_exp_sigma_fields`、`_exp_default_curve_fields{,_v2}` 三个实验场缓存，gitignored；`data/_dprime_generator.py` 自检②按新路径引用）。
- **logs/**（7.8GB，gitignored）：training_history 22 目录（**v5b_s3/s4/s5 收官产物 + v5a6_dprime_* 历史 + v5a6_demo 汇报件全保留**；2026-10-01 已清第一档 12 目录 ~3.2GB：cli_smoke/cli_test/cli_test2/run_train_v5/smoke_v5a6/v5a5_smoke{,_resume}/v5a6_smoke/resnet18_150ep/v5a6_overfit32/_discard_staleGT/_v5028_pre_regen_bak + sim_cache/_backup_badkpts）+ sim_cache 869MB（新 dpgs 键 + 旧键并存——旧键被归档评估脚本引用，约束保留）。
- **_backups/**（1.1GB，未跟踪）：用户手工备份，勿动。

## 遗留待办

1. **判据 3 评估口径修正（P1）**：过夜 `_eval_s3_*.py` 误用同输入重复前向 → 修正后重测干净 GT 基线（立项文档 §六.3）
2. **V5-034 idx=101 GT 骨架崩塌（P2）**：不阻塞 v5-β
3. **`_diag_aug_marker.py` 转正式单测**（立项文档 §四建议，转正时一并改名 test_*.py）
4. **scripts/_archive/ 归档件复跑**：`_exp_v5025_cause.py` 等硬编码旧 `_exp_sigma_fields` 路径——复跑需先按 `output/exp_fields/` 新家改
5. **_backups/ 1.1GB**：用户备份，是否瘦身由用户决定

## 关联资产

| 资产 | 路径 | 关系 |
|---|---|---|
| 家族根索引 | `../PROJECT_INDEX.md` | 演进全景 |
| v4 | `../project_v4/` | fork 源（legacy_v4/ 件逐字节同源） |
| 演进史总领份 | `D:/New_team_project/docs/knowledge_base/pe_mmnet演进史.md` | 唯一编辑入口（`sync_evolution.sh` 单向同步到 docs/ 项目份） |
| new_project2316 bridge | `../../new_project2316/system/pe_mmnet_bridge/` | sys.path 依赖本目录（thermal/gp/spline/loss 四桥） |
