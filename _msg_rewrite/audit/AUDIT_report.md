# AUDIT_report.md — msg-rewrite 六项独立审计（2026-10-05）

**范围**：`8b85ab9..msg-rewrite` 45 条。**环境**：msg-rewrite=7b31349=origin/msg-rewrite；
本地 main=be7a5c2（=origin/main c66f90b + 快照 commit 未推）；main 树=7b31349 树=c5a6480 树=`91079c15`。
**总判定：审计不通过——审计 6 揪出重大事故：批次 3 的 12 条改写在批次 4 重建时被静默回退为 main 原文。**
完好链 c5a6480 仍在（仅 reflog 可达）。证据如下，每项=命令+原始输出+判定。

> **附二（2026-10-05 追加）：committer date 已恢复 main 原值。**
> 审计揪出的事故经阶段 2（c5a6480 延伸重建，44!+1= 独立 oracle 全过）与阶段 3
> （git filter-repo --commit-callback 按 _mr_to_cdate.tsv 恢复 committer date）修复。
> 最终链 `msg-rewrite-b3-plus-c5a6480` = `cf83c1c` = origin/msg-rewrite；
> 三重校验：`git diff main <链>` = 0 字节、author date 逐条 diff 空、
> **committer date 逐条 diff 空（本次修复目标，已达成）**；
> message 逐条对 main 仍 44 条不同（改写完整保留），树 hash `91079c15` 与 main 一致。
> 事故链存档：msg-rewrite(7b31349)=事故现场、backup-msg-rewrite-b3(c5a6480)=39 条改写完好链、
> backup-msg-rewrite-v2(aeca8c5)=committer 恢复前中间版。

---

## 审计 1：hash 引用全部 main 祖先
命令：45 条 message 提取 unique hex，逐个 `git merge-base --is-ancestor <h> main`。
输出（`_msg_rewrite/_audit_raw.txt`）：22 个 unique hex 逐个打印，末行 `小计: 22/22 ancestor`，无 NOT-ANCESTOR/NOT-RESOLVABLE。
判定：**PASS**（含 429a050@pos44、e5cf806@pos43 等后向跨位引用，全部合法）。

## 审计 2：无前向引用
命令：main 位置表 vs 分支每条 message 内 hex 位置比对。
输出：`FORWARD: 73b1a99@main-pos16 出现在分支 pos16`＋计数 1。
原文（分支 pos16 [验证] 段）：`[验证] git show 73b1a99 单文件 1 行变更；…`
判定：**PASS**——唯一计分是条目引用自身 main hash（同位自引，非指涉他条前向）。是否豁免请裁决。

## 审计 3：[背景] 禁语 grep
命令：`前一条|上一条|上一 commit|前一 commit` 对分支 message＋全部源 .txt 双查。
输出：`命中: 分支 message 0 处，源 .txt 0 处`；弱变体披露 1 处：
`pos40 9ffccbd: …反弹（0.829/0.742）。前条落档的解读"任意 20% 邻域`（P3b 条目，[背景] 首句是问题陈述，"前条"为段中叙事衔接）。
判定：**PASS，弱变体 1 处待裁决**。

## 审计 4：关键数字溯源（用户指定 5 锚点 7 数字）
命令：message 含数字＋源文档 grep 原句，双查。输出：
```
PASS 7497150→a9c495 "7.05"/"0.43"  源: CHANGELOG.md 判据 4 ratio **7.05→0.43 PASS**（阈值 1.5）
FAIL 90a0ae8→60bdd3 "11/200"      message含=False; 源: CHANGELOG.md …GT 恰 11/200 不同…
PASS 0072353→768bfb "0.1191"      源: CHANGELOG.md inter_var=0.1191 PASS（阈 ≥1e-3，119 倍余量）
PASS 643d1fb→9ffccb "0.2450"/"0.8287" 源: docs/mc_paper_readiness.md warm max=0.2450，是冷启动 0.8287 的 -70%
PASS b21c5c1→615c8b "0.330"       源: docs/mc_paper_readiness.md 5 轮 max 0.843/0.279/0.307/0.325/0.330
```
唯一 FAIL 的定性：`git log --format=%B -1 6127d44 | grep -c 11/200` → `1`（6127d44=90a0ae8 的批次 3 改写，在 c5a6480 链 pos30）。
判定：**7 个数字全部真实可溯源、无一编造；当前分支 6/7 在 message 内，缺的 1 个是审计 6 事故所致**。

## 审计 5：作废/推翻/撤回交叉标注
输出：24 行命中（`_audit_raw.txt` 全文）。主要双向对：pos11→12（V5-017 推翻）、
pos19→20（初版结论推翻）、pos24/25→26（rot90 污染作废五~九轮结论）、pos21（32×32 撤回依据）、pos38/39→40（P3b 撤回强宣称）。
判定：**PASS**——每处撤回都有对象条目＋撤回声明双向在场。

## 审计 6：五段式完备性 ← **事故暴露点**
输出：`完整四段: 32/45`；不完整=pos26-37 共 12 条四段全缺＋pos45（快照，设计内缺[实现/修改]）。
坐实命令与输出：
```
pos26: main=7497150 vs 分支=a9c4952 -> message 完全一致（改写丢失）
pos30: main=90a0ae8  vs 分支=60bdd3c -> message 完全一致（改写丢失）
pos37: main=ce8aab8  vs 分支=ba34cfb -> message 完全一致（改写丢失）
pos26-37 与 main 逐字节相同: 12/12
```

## 事故根因（三层）
1. **直接原因**：`batch4/_rebuild.sh` L41-52 把批次 3 十二条标为不改写——
   `replay 7497150 2026-10-01 "" 0  # pos26 未改写,保main原message`（12 行同型，L41-52）。
2. **上游原因**：`gen_plan.py` L38 `DEFAULT_DONE` 只含批次 1+2 共 27 个 hash；
   批次 3 的 12 个 hash 逐个 grep 全部"不在"。批次 4 生成时未传修正的 --done。
3. **防线失守**：gen_plan.py 预期行按 DEFAULT_DONE 打印 27+5=**32** 条 ！，
   实测 32!+13= 与之吻合 → 当作 PASS。**对照的是脚本自己的错误预期，不是独立账目
   44!+1=（b1 10+b2 17+b3 12+b4 5=44 改写，pos45 快照除外）**。树 hash/author date/
   committer date 三重校验按设计不查 message 内容，拦不住此事故。

## 损伤与完好链
- 损伤：msg-rewrite（本地＋origin）=7b31349，批次 3 十二条改写回退为 main 原文。
- 完好链 **c5a6480**（批次 3 收官推送版）四项复核全过：
  `range-diff 对 main = 39!+6=`（=为批次 4 五条当时未改写＋pos45）；树=`91079c15` 与 main 一致；
  author date 逐条 diff 空；`c5a6480 链 pos26-37` 即 d5e419e/9305fcf/5e05deb/0fae065/6127d44/76d1507/
  d7046f4/421bc8c/b6e3f58/c292e0c/e7997b8/adb2d8e 十二条改写全在场（含 11/200、0.1191）。
- `git branch -a --contains c5a6480` → **空**（无分支包含，仅 reflog 4 处可达，GC 保护约 30-90 天）。
- 批次 4 五条源 .txt vs 当前分支 message：5/5 一致（仅尾随换行差）——无需重起草。

## 建议修复（未执行，等裁决）
1. `git branch backup-msg-rewrite-b3 c5a6480`（先保护完好链，防 reflog 过期）。
2. `gen_plan.py` DEFAULT_DONE 补批次 3 十二个 hash（或 --done 显式传 39 个）。
3. 重新生成 batch4 PLAN → dry-run `tail` 验 pos26-37 变为 amend 行 → 执行 →
   校验对**独立 oracle 44!+1=** → 推送。
4. 通过后按原指令执行 committer date 恢复（方案不变）。

## 本报告边界
未 commit、未 push、未建备份分支、未动 main、未动 msg-rewrite、未改 gen_plan.py。
原始输出全文：`_msg_rewrite/_audit_raw.txt`（脚本 `_audit.py` 可复跑）。
