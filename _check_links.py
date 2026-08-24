import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
broken = []
total = 0

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

for root, _, files in os.walk(os.path.join(PROJECT_ROOT, 'docs')):
    for f in files:
        if not f.endswith('.md'):
            continue
        path = os.path.join(root, f)
        with open(path, 'r', encoding='utf-8') as fp:
            content = fp.read()
        for ln, text in enumerate(content.splitlines(), 1):
            for m in re.finditer(r'\]\(([^)]+)\)', text):
                link = m.group(1).split('#')[0].split('|')[0]
                if not link.endswith('.md') or link.startswith('http'):
                    continue
                total += 1
                base = os.path.dirname(path)
                target = os.path.normpath(os.path.join(base, link))
                if not os.path.exists(target):
                    rel = os.path.relpath(path, PROJECT_ROOT)
                    rel_t = os.path.relpath(target, PROJECT_ROOT)
                    broken.append(f'{rel}:{ln}: {rel_t}')

print(f'Total .md links: {total}')
print(f'Broken links: {len(broken)}')
for b in broken:
    print(f'  BROKEN: {b}')
sys.exit(1 if broken else 0)