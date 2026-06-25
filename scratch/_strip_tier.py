"""One-off: strip every `tier=...` kwarg from Metric() call sites (note 9).

Handles three forms: a tier= arg alone on its own line, an inline tier= sharing
a line with other args, and a trailing tier= (comma before). Does NOT touch
bench_metrics.py (the Metric definition / range fields are edited by hand).
"""
import re
import glob
import os

BASE = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\src\oculomotor\benchmarks'
VAL = r"(?:'[^']*'|\"[^\"]*\"|\([^()]*\))"   # quoted string or single-level paren expr

drop_line = re.compile(r"(?m)^[ \t]*tier\s*=\s*" + VAL + r"\s*,\s*\n")
inline    = re.compile(r"\btier\s*=\s*" + VAL + r"\s*,[ \t]*")
trailing  = re.compile(r",[ \t]*\btier\s*=\s*" + VAL)

total = 0
for path in sorted(glob.glob(os.path.join(BASE, 'bench_*.py'))):
    if os.path.basename(path) == 'bench_metrics.py':
        continue
    src = open(path, encoding='utf-8').read()
    n0 = src.count('tier=')
    src = drop_line.sub('', src)
    src = inline.sub('', src)
    src = trailing.sub('', src)
    n1 = src.count('tier=')
    if n1 != n0:
        open(path, 'w', encoding='utf-8').write(src)
        print(f'  {os.path.basename(path):22s} {n0} -> {n1}')
        total += (n0 - n1)
print('stripped', total, 'tier= occurrences')
