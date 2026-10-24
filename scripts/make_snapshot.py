#!/usr/bin/env python3
"""Embed a snapshot of the campaign state (and of the live tree) in the site page,
for viewers who cannot read the project files. Run before each publication."""
import json, re, sys
ROOT = '/mnt/project-files/nug30'
page = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/site/nug30_lab.html'
etat = json.load(open(f'{ROOT}/etat/etat.json'))
J = [j for j in etat['jobs'] if '-stats ' in (j.get('args') or '')]
job = next((j for j in J if j['status'] == 'running'), None) or \
      max((j for j in J if j['status'] == 'done'), key=lambda j: j.get('end') or 0, default=None)
snap = {'etat': etat}
if job:
    snap['job'] = job['id']
    for k, f in (('stats', 'stats.json'), ('progress', 'progress.json')):
        try: snap[k] = json.load(open(f'{ROOT}/runs/{job["id"]}/{f}'))
        except Exception: pass
txt = json.dumps(snap, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
s = open(page).read()
s2, n = re.subn(r'<script type="application/json" id="snap">.*?</script>',
                lambda m: f'<script type="application/json" id="snap">{txt}</script>', s, count=1, flags=re.S)
assert n == 1
open(page, 'w').write(s2)
print(f'snapshot: {len(txt)} bytes, job {snap.get("job")}')
