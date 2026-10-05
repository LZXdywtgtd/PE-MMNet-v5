MAPPING = {}
with open(r'D:/New_team_project/projects/pe_mmnet/project_v5/_mr_to_cdate.tsv',
          encoding='utf-8') as f:
    for line in f:
        k, v = line.split()
        MAPPING[k] = v

from datetime import datetime
oid = commit.original_id.decode()
if oid in MAPPING:
    dt = datetime.fromisoformat(MAPPING[oid])
    commit.committer_date = ('%d %s' % (int(dt.timestamp()), dt.strftime('%z'))).encode()
