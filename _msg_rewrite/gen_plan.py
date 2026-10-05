#!/usr/bin/env python3
"""重建 PLAN 生成器——commit message 重写分支的脚本化工具。

规则（用户裁决 2026-10-05）：
  重建 PLAN 的 hash 序列必须由 git log --reverse 程序化生成，
  不允许手工抄写。本脚本从 git 生成 replay 脚本并内置三项自检，
  输出需人工 review 后才能执行。

用法：
  python _msg_rewrite/gen_plan.py --batch batch3 \
      --scope _msg_rewrite/batch3/_scope.txt

_scope.txt：本批要改写的 main hash，每行一个（# 注释）。

生成逻辑：
  - 序列来源：git rev-list --reverse <base>..<source>，唯一权威
  - 位置对齐：main 列表与 msg-rewrite 分支列表按位置对齐，
    逐位比对树 hash（不一致立即报错——批次2事故的根因检查）
  - 已改写条目（批次1/2）：用分支 hash 重放 + --no-edit（天然
    保留新 message，杜绝被 main 旧 message 还原）
  - 本批条目：用 main hash 重放 + amend batchN/<hash>.txt
  - 未改写尾部：用 main hash 重放 + --no-edit
  - 尾部落分支：git branch -f <branch> HEAD + checkout（脚本只
    重放到 detached HEAD 不落分支，批次 3 事故教训）

输出：
  _msg_rewrite/<batch>/_rebuild.sh   （执行前必须人工 review）
  stdout                              （review 清单）
"""
import argparse
import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 已改写批次的 main hash 集合（批次1-4 全量 44 条）。可用 --done 覆盖。
# 警示：本表曾因只含批次 1+2 的 27 个 hash 导致批次 4 重建静默回退批次 3
# 十二条改写（六项审计揪出，证据链 _msg_rewrite/audit/）。补全后仅为历史
# 参考保留——批次 4 已终，勿再重跑重建；若确需重建，改写集合必须以
# batch*/*.txt 文件清点为准，不得信赖任何硬编码清单。
DEFAULT_DONE = """50fb408 429a050 9d6e1aa 47be2ec 5f671cf 73b1a99
8856c10 39b6ad3 19ea6ba c66f90b
9c28ac3 9f407a3 9dabfae c28d99f f51554c 42c9296 53dfa68 c548667
54512fe 96e2ef0 5245d2b 65a8362 60fbc34 f2155dd 121636d 29912fa
440d139 7497150 620fc6a ceefa2e 2829242 90a0ae8 b475013 c6e8e1e
be424d6 9562f7c 0072353 b49d89e ce8aab8
e5cf806 a30078d 643d1fb ea5e198 b21c5c1""".split()


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} 失败:\n{r.stderr}")
    return r.stdout


def short(lst):
    return [h[:7] for h in lst]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="8b85ab9")
    ap.add_argument("--source", default="main")
    ap.add_argument("--branch", default="msg-rewrite")
    ap.add_argument("--batch", required=True, help="batch3 / batch4 ...")
    ap.add_argument("--scope", required=True, help="本批 main hash 清单文件")
    ap.add_argument("--done", default="", help="已改写 main hash 清单文件（空=内置批次1+2）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = git("rev-parse", "--show-toplevel").strip()
    os.chdir(root)

    # ---- 序列：唯一权威来源 ----
    M = short(git("rev-list", "--reverse", f"{args.base}..{args.source}").split())
    B = short(git("rev-list", "--reverse", f"{args.base}..{args.branch}").split())
    if len(M) != len(B):
        sys.exit(f"FATAL 两分支长度不一致 main={len(M)} {args.branch}={len(B)}")

    # ---- 位置对齐：逐位树 hash 比对（批次2事故根因检查）----
    mis = [i for i, (m, b) in enumerate(zip(M, B))
           if git("rev-parse", f"{m}^{{tree}}").strip() !=
              git("rev-parse", f"{b}^{{tree}}").strip()]
    if mis:
        for i in mis:
            print(f"FATAL 位置 {i+1} 树不一致 main:{M[i]} branch:{B[i]}")
        sys.exit("位置对齐失败——禁止生成 PLAN")

    # ---- 范围 ----
    with open(args.scope, encoding="utf-8") as f:
        scope = [l.strip() for l in f
                 if l.strip() and not l.strip().startswith("#")]
    done = short(open(args.done, encoding="utf-8").read().split()) \
        if args.done else DEFAULT_DONE
    unknown = [h for h in scope if h not in M]
    if unknown:
        sys.exit(f"FATAL scope 中不是 {args.source} 祖先: {unknown}")

    # ---- .txt 存在性：本批条目必须有其目录下的 .txt ----
    missing = [h for h in scope
               if not os.path.exists(f"_msg_rewrite/{args.batch}/{h}.txt")]
    if missing:
        sys.exit(f"FATAL 缺 message 文件: {missing}")

    # ---- 已改写条目：定位其 .txt（batch1/batch2），缺失则报错 ----
    # 源 .txt 是唯一权威——重建时一律 amend，修正（如前向引用清理）
    # 随重建自动生效，而不是依赖分支上的旧 message
    done_txt = {}
    for h in done:
        if h not in M:
            continue
        found = [d for d in ("batch1", "batch2", args.batch)
                 if os.path.exists(f"_msg_rewrite/{d}/{h}.txt")]
        if not found:
            sys.exit(f"FATAL 已改写条目 {h} 缺 .txt（batch1/batch2/{args.batch} 均无）")
        done_txt[h] = found[0]

    dates = dict(zip(M, git("log", "--reverse", "--format=%ad",
                            "--date=short", f"{args.base}..{args.source}").split()))
    pos = {h: i + 1 for i, h in enumerate(M)}

    # ---- 生成 ----
    lines = ["#!/bin/bash", "set -e",
             f"cd /d/New_team_project/projects/pe_mmnet/project_v5",
             f"git checkout -q {args.base}", "",
             'replay() {',
             '  local oldh="$1" date="$2" msgf="$3" rewrite="$4"',
             '  git cherry-pick "$oldh" >/dev/null',
             '  if [ "$rewrite" = "1" ]; then',
             '    GIT_COMMITTER_DATE="$date 12:00:00 +0800" git commit -q --amend -F "_msg_rewrite/$msgf"',
             '  else',
             '    GIT_COMMITTER_DATE="$date 12:00:00 +0800" git commit -q --amend --no-edit',
             '  fi', '}', ""]
    for m, b in zip(M, B):
        d = dates[m]
        if m in scope:
            lines.append(f"replay {m} {d} {args.batch}/{m}.txt 1  # pos{pos[m]} 本批改写")
        elif m in done:
            lines.append(f"replay {m} {d} {done_txt[m]}/{m}.txt 1  # pos{pos[m]} 已改写,amend源txt")
        else:
            lines.append(f"replay {m} {d} \"\" 0  # pos{pos[m]} 未改写,保main原message")
    lines += ["echo DONE", "git rev-parse HEAD",
              f"git branch -f {args.branch} HEAD",
              f"git checkout -q {args.branch}", ""]

    out = args.out or f"_msg_rewrite/{args.batch}/_rebuild.sh"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    n_batch = len(scope)
    print(f"已生成 {out}")
    print(f"review 清单：")
    print(f"  总条目 {len(M)}；本批改写 {n_batch} 条（pos "
          f"{sorted(pos[h] for h in scope)}）；已改写 amend源txt "
          f"{sum(1 for m in M if m in done)}；未改写 "
          f"{sum(1 for m in M if m not in scope and m not in done)}")
    print(f"  树对齐：{len(M)}/{len(M)} 位一致")
    print(f"  执行后校验预期：range-diff 恰 {sum(1 for m in M if m in done) + n_batch} 条 !"
          f"（累计改写=已改写 {sum(1 for m in M if m in done)} + 本批 {n_batch}）；"
          f"树 hash = git rev-parse {args.source}^{{tree}}")


if __name__ == "__main__":
    main()
