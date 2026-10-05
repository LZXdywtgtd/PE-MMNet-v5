# commit message 重写——批次流程规则

> 批次 2 事故与用户裁决后固化（2026-10-05）。批次 3/4 执行前必读。

## 已定惯例（硬规则）

### 1. 引用规则（2026-10-05 用户裁决）

- message 里**只允许后向 hash 引用**（引用历史中更早的 commit），**保留 main 历史 hash**——同仓 `git show <main-hash>` 任何分支可解析，天然自洽
- **前向引用一律删除 hash，改描述性文字**（"后续 42c9296 复核"→"后续默认曲线复核"）。依据：写 message 时未来还没发生，前向引用本不该存在
- **hash 引用必须在 main 上可解析**：起草时对每个引用跑 `git merge-base --is-ancestor <hash> main`，返回非 0 即违规（防止误用 msg-rewrite 上的新 hash——新 hash 在重建后漂移成孤儿）
- 由此不需要跨批 hash 一致性检查，不需要为 message 维护新旧 hash 对照表

### 2. [背景] 硬规则（2026-10-04 用户警告 + 2026-10-05 补充，批次 3 尤其）

- [背景] **不允许写成"上一条 commit X 做了 Y"**——必须写成"上游问题是什么"
- **[背景] 首句不允许以"前一条 commit 做了 X"开头**
- 引用来源优先级：CHANGELOG 版本条目（v5.0.15~v5.0.18）> 已知问题条目 > 演进史 §四 > 本条 commit 的 diff
- **相邻 commit 的 message 不作为 [背景] 素材来源**
- 来源标注 A/B/C/D 四类纪律照旧（A 原文已有/B diff 文件列表/C 已有文档原句/D 定性判断标依据）

### 3. 重建必须脚本化（2026-10-04 用户指令 + 2026-10-05 修正）

- PLAN 由 `_msg_rewrite/gen_plan.py` 生成（`git rev-list --reverse` 唯一权威），**禁止手工抄 hash**
- 生成器内置：位置对齐树 hash 逐位比对（批次 2 事故根因检查）、scope/done 校验、.txt 缺失拦截
- 已改写条目（batch1/batch2）与 本批条目**一律 amend 源 .txt**——源 .txt 是唯一权威，后续修正（如前向引用清理）随重建自动生效
- **修改 .txt 后不重建不生效**：分支上的 message 不会因 .txt 改动而更新。改完 .txt 必须重跑 `gen_plan.py` → review 脚本 → 重建 → 强推，三者（源 .txt / 分支 / 权威状态）才重新同步。改了 .txt 以为立即生效是最容易踩的坑
- 生成后人工 review 脚本，才能执行
- 生成的脚本含**尾部落分支**（`git branch -f <branch> HEAD`+checkout）——脚本只重放到 detached HEAD，不落分支（批次 3 事故：重放完直接校验，读到的还是旧分支）

### 3b. 重建流程显性 checklist（2026-10-05 用户指令，批次 3 事故后固化）

> 教训：批次 2 手工流程是对的（补了分支更新），脚本化时把隐式步骤漏了——"脚本化"是有损转换。每步必须显性化并带验证。

| # | 步骤 | 手工版 | 脚本版 | 怎么验证没漏 |
|---|---|---|---|---|
| 1 | PLAN 生成 | 手抄 hash（**已禁止**，批次 2 逆序事故） | `gen_plan.py --batch batchN --scope ...` | review 清单"树对齐 45/45"；脚本行数=45+函数定义 |
| 2 | 尾部落分支 | 手工 `git branch -f msg-rewrite HEAD`（批次 2 补过，脚本化时曾漏） | 模板尾部 `git branch -f` + `checkout` | **生成后 dry-run**：`tail _rebuild.sh` 实见两行才执行，不信"我已改模板" |
| 3 | committer date | `GIT_COMMITTER_DATE="日期 12:00:00 +0800"` | replay() 内置 | 执行后逐条 `%cd` 全部 12:00:00 |
| 4 | author date 保留 | cherry-pick 自动保留（批次 1/2 曾有 10 条被误写成 12:00:00，抽样没抓住） | cherry-pick 自动保留 | **逐条硬校验，不抽样**：`diff <(git log --reverse --format='%aI' 8b85ab9..main) <(git log --reverse --format='%aI' 8b85ab9..<branch>)` 无输出 |
| 5 | 树 hash 一致 | — | — | `git rev-parse main^{tree}` == `git rev-parse <branch>^{tree}` |
| 6 | range-diff 对数 | — | — | **对"累计改写数"**（已改写+本批，生成器末行打印），不是本批数；`=` 数=未改写条数 |
| 7 | 强推 | `git push --force origin <branch>:refs/heads/msg-rewrite` | 同左 | 推送输出旧→新 tip 与校验用 tip 一致 |

**校验读错对象是批次 3 事故的直接形态**：分支没更新时，所有校验读的是旧分支——数字可能"看起来合理"（27 条 ! = 批次 1+2 改写数），只有知道预期总数（39）才能识破。

### 4. 校验与推送

- 三重校验（**author date 逐条 diff，不抽样**——批次 1/2 的"逐条保留"声称被批次 3 逐条 grep 证伪过 10 条）：
  - `git rev-parse main^{tree}` == `git rev-parse <branch>^{tree}`
  - range-diff `!` 条数 = 累计改写数（batch1+batch2+本批），`=` 条数 = 未改写数
  - `diff <(git log --reverse --format='%aI' 8b85ab9..main) <(git log --reverse --format='%aI' 8b85ab9..<branch>)` 无输出；committer 逐条 12:00:00
- `git push --force origin <local-branch>:refs/heads/msg-rewrite`
- 批间流程：推分支 → 等用户 review → 下一批

### 5. committer date 恢复时机（2026-10-05 用户指令）

- 批次间**不要恢复** committer date（当日 12:00 惯例）——批次 4 会再次重建，恢复会白做
- **批次 4 完成后一次性恢复全部 58 条**的真实 committer date（author date 全程不动）

## 批次范围（已确认）

- 批次 3（12 条，10-01~10-02）：`_msg_rewrite/batch3/_scope.txt`
- 批次 4（5 条，10-03~10-04）：`_msg_rewrite/batch4/_scope.txt`
- 88a85ce 不改（保留作样本）

## 批次 3 特别警示

12 条高度耦合（rot90 修复→口径分离→过夜脚本→落档→缓存键根治→结构重组→v5b-pre 清偿链），[背景] 最容易退化成前一条复述。每条 [背景] 必须从 CHANGELOG/已知问题/演进史/diff 里找"上游问题"，写问题不写"上一条干了什么"。

**c6e8e1e（结构重组）单列警示**：12 条里唯一既非修复也非落档的 commit，最容易写成流水账。其 [背景] 应是"v5-α 收官后根目录堆积约 90 个文件，需按'活不活'分类归位"（素材：CHANGELOG v5.0.17 条目），而不是"前面几条 commit 做了 X 所以要整理"。

## 事故保护机制（2026-10-05 收官后固化）

> 依据：批次 3 改写丢失事故（六项审计揪出，证据链在 `audit/msg-rewrite-incident` 分支 =96569e4，
> 含 AUDIT_report.md、审计脚本、原始输出、映射 TSV）。未来同类历史重写工程照此模式执行。

**模式：tag + archive/ 分支 + 双端，三重保护并存。**

1. **重写操作前先建 backup 分支**——动任何已有链之前，把当前顶点固化为引用（本轮：
   `backup-msg-rewrite-b3`→c5a6480 完好链、`backup-main-prerewrite`→be7a5c2 旧 main）。
   只有 reflog 可达的链约 30-90 天会被 GC，**等事故确认再建分支就晚了**。
2. **tag 与 archive/ 分支并存，不是二选一**：
   - branch 可被 `git branch -d` 顺手删掉、checkout 覆盖；tag 语义强（"历史存档"）、防误删。
     两者指向同一 hash，成本为零，语义互补。
   - 命名：事故/中间态用 `archive/` 前缀分支（`archive/msg-rewrite-7b31349` 事故现场、
     `archive/msg-rewrite-aeca8c5` 中间态）；重写前后基线用 `pre-rewrite-*` tag。
3. **证据 commit 落独立分支，绝不落在事故现场链上**——`audit/*` 分支专属证据，
   事故现场分支（archive/msg-rewrite-7b31349）保持纯净（本轮曾把证据 commit 误落
   msg-rewrite 上，用户裁定单开 audit 分支 + reset 还原现场）。
4. **全部引用推远端**——只在本机的"保护"在磁盘故障面前等于零。验证方式：
   `git ls-remote origin` 看引用在列即远端可解析（收官验证 2026-10-05：
   `refs/tags/pre-rewrite-main-origin`→c66f90b 在列）。
5. **冗余清理在双端保护确认之后**——同指向的重复分支（如 archive/msg-rewrite-final-cf83c1c
   与 main 同内容）先推后删；origin 上已无活跃开发的同名分支（origin/msg-rewrite）删掉
   防止误以为仍有开发。

**本轮终态引用清单**（2026-10-05，全部双端）：main=aa4bf8a｜audit/msg-rewrite-incident=96569e4｜
backup-main-prerewrite=be7a5c2｜archive/msg-rewrite-{7b31349,aeca8c5,c5a6480}｜
tag：pre-rewrite-main-{origin c66f90b,local be7a5c2}、archive-{b3-c5a6480,aeca8c5,incident-7b31349}。

**关联规则**：校验必须对独立 oracle（memory 六硬规则第 6 条）是这套保护机制存在的原因——
三重校验（树/author date/committer date）按设计不查 message 内容，拦不住"改写静默回退"，
只有硬编码清单与独立清点对不上时审计才能揪出。
