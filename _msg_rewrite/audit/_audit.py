#!/usr/bin/env python3
"""六项审计——msg-rewrite 分支 message 独立核查（外部 review 用，2026-10-05）。
每节打印核心命令与原始输出；判定只基于本脚本 stdout。"""
import subprocess, re, sys, glob, os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(r'D:\New_team_project\projects\pe_mmnet\project_v5')

def git(*a):
    return subprocess.run(['git', *a], capture_output=True, encoding='utf-8',
                          errors='replace')

R = git('log', '--reverse', '--format=%H%x00%B%x1e', '8b85ab9..msg-rewrite').stdout
msgs = []
for e in R.split('\x1e'):
    if e.strip():
        h, b = e.split('\x00', 1)
        msgs.append((h, b.rstrip('\n')))
M = git('log', '--reverse', '--format=%H', '8b85ab9..main').stdout.split()
mpos = {h: i + 1 for i, h in enumerate(M)}
mpos7 = {h[:7]: i + 1 for i, h in enumerate(M)}
bpos = {h[:7]: i + 1 for i, (h, _) in enumerate(msgs)}
m2b = dict(zip([h[:7] for h in M], [h[:7] for h, _ in msgs]))
tip = git('rev-parse', '--short', 'msg-rewrite').stdout.strip()
otip = git('rev-parse', '--short', 'origin/msg-rewrite').stdout.strip()
print(f'环境：msg-rewrite={tip} origin/msg-rewrite={otip}；'
      f'范围 8b85ab9..msg-rewrite 共 {len(msgs)} 条（main 同范围 {len(M)} 条）')

HEX = re.compile(r'(?<![0-9a-f])[0-9a-f]{7,9}(?![0-9a-f])')

print('\n### 审计1: hash 引用全部 main 祖先')
print('命令: 对每条 message 提取 unique 7-9 位 hex，逐个 git merge-base --is-ancestor <h> main')
hexes = {}
for h, b in msgs:
    for m in HEX.finditer(b):
        hexes.setdefault(m.group(0), []).append(bpos[h[:7]])
n_ok = 0
for hx in sorted(hexes):
    rc = git('merge-base', '--is-ancestor', hx, 'main').returncode
    st = {0: 'ancestor', 1: 'NOT-ANCESTOR'}.get(rc, 'NOT-RESOLVABLE(rc=%d)' % rc)
    if rc == 0:
        n_ok += 1
    print(f'  {hx}  @pos{hexes[hx]} -> {st}')
print(f'  小计: {n_ok}/{len(hexes)} ancestor')

print('\n### 审计2: 无前向引用（引用位置号 < 本条位置号）')
bad = 0
for h, b in msgs:
    bp = bpos[h[:7]]
    for m in HEX.finditer(b):
        hx = m.group(0)
        p = mpos7.get(hx)
        if p is None:
            p = next((v for k, v in mpos.items() if k[:len(hx)] == hx and len(hx) >= 7), None)
        if p is not None and p >= bp:
            print(f'  FORWARD: {hx}@main-pos{p} 出现在分支 pos{bp}')
            bad += 1
print(f'  前向引用计数: {bad}')

print('\n### 审计3: [背景] 禁语 grep')
print('命令: grep -n "前一条|上一条|上一 commit|前一 commit"（分支 message + 源 .txt 双查）')
pat = re.compile(r'前一条|上一条|上一 commit|前一 commit')
hits = 0
for h, b in msgs:
    for line in b.splitlines():
        if pat.search(line):
            print(f'  分支 pos{bpos[h[:7]]} {h[:7]}: {line.strip()[:60]}')
            hits += 1
t2 = 0
for f in glob.glob('_msg_rewrite/batch*/*.txt'):
    if not re.fullmatch(r'[0-9a-f]{7}\.txt', os.path.basename(f)):
        continue
    for i, line in enumerate(open(f, encoding='utf-8'), 1):
        if pat.search(line):
            print(f'  {f}:{i}: {line.strip()[:60]}')
            t2 += 1
print(f'  命中: 分支 message {hits} 处，源 .txt {t2} 处')
print('  补充披露（弱变体，不在指定 grep 内，供人工判断）:')
soft = 0
for h, b in msgs:
    for line in b.splitlines():
        if re.search(r'前条|前述|前面几条', line):
            print(f'  pos{bpos[h[:7]]} {h[:7]}: {line.strip()[:60]}')
            soft += 1
print(f'  弱变体命中: {soft} 处')

print('\n### 审计4: 关键数字可溯源（message 与源文档双查）')
targets = [('7497150', ['7.05', '0.43']), ('90a0ae8', ['11/200']),
           ('0072353', ['0.1191']), ('643d1fb', ['0.2450', '0.8287']),
           ('b21c5c1', ['0.330'])]
docs = ['CHANGELOG.md', 'docs/mc_paper_readiness.md', 'docs/pe_mmnet演进史.md']
n_pass = 0; n_total = 0
for main_h, nums in targets:
    bh = m2b[main_h]
    body = dict(msgs)[[h for h, _ in msgs if h[:7] == bh][0]]
    for num in nums:
        n_total += 1
        inmsg = num in body
        src = ''
        for d in docs:
            for line in open(d, encoding='utf-8'):
                if num in line:
                    src = f'{d}: {line.strip()[:66]}'
                    break
            if src:
                break
        ok = inmsg and bool(src)
        n_pass += ok
        print(f'  {"PASS" if ok else "FAIL"} {main_h}(分支{bh}) "{num}": message含={inmsg}; 源: {src or "未找到"}')

print('\n### 审计5: 作废/推翻/撤回 交叉标注')
n5 = 0
for h, b in msgs:
    for line in b.splitlines():
        if re.search(r'作废|推翻|撤回', line):
            print(f'  pos{bpos[h[:7]]} {h[:7]}: {line.strip()[:72]}')
            n5 += 1
print(f'  含作废/推翻/撤回语句的行: {n5} 行')

print('\n### 审计6: 五段式完备性')
full = 0; missing = []
for h, b in msgs:
    segs = {'背景': '[背景]' in b, '实现/修改': '[实现]' in b or '[修改]' in b,
            '验证': '[验证]' in b, '已知问题': '[已知问题]' in b}
    if all(segs.values()):
        full += 1
    else:
        missing.append((bpos[h[:7]], h[:7], [k for k, v in segs.items() if not v]))
print(f'  完整四段([背景]+[实现/修改]+[验证]+[已知问题]): {full}/{len(msgs)}')
for p, h, miss in missing:
    print(f'  不完整: pos{p} {h} 缺 {miss}')
