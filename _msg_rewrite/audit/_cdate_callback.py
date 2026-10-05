#!/usr/bin/env python3
"""阶段3 committer date 恢复回调（git filter-repo --commit-callback 用）。
按 _mr_to_cdate.tsv（original_id=main 40位 hash -> main 原 committer date）
恢复 msg-rewrite-b3-plus-c5a6480 链上 45 条的 committer date。
不在表中的 commit 保持原值（回调范围已用 --refs 限定为该分支，全部 45 条应在表中）。"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MAPPING = {}
with open(r'D:\New_team_project\projects\pe_mmnet\project_v5\_mr_to_cdate.tsv',
          encoding='utf-8') as f:
    for line in f:
        k, v = line.split()
        MAPPING[k] = v

oid = commit.original_id.decode()
if oid in MAPPING:           # 表内恢复；表外=8b85ab9 之前祖先，原样放过
    commit.committer_date = MAPPING[oid].encode()
