"""Honest check on the re-selected ensemble.

Greedy forward selection on five folds will always look good on those same five folds.  This asks
how much of the 46.11 is real by measuring the ensemble on a protocol the selection never saw:

  leave-one-fold-out selection -- pick the members using four folds, score on the fifth
  vs
  select on all five folds, score on the fifth (the optimistic number)

If the two are close, the selection generalises.  If the optimistic number is much better, the
46.11 is selection noise and should be reported as such.
"""
import glob
import os
import sys

import numpy as np

OUT = os.environ.get('Q2_OUT', 'runs')
NAMES = sys.argv[1:] or ['g3_l3k97', 's3_v1', 's1_e25', 'n1_dk97']


def load(name):
    per = {}
    for f in sorted(glob.glob(f'{OUT}/{name}_f*_s*.npz')):
        d = np.load(f, allow_pickle=True)
        per.setdefault(int(d['fold']), []).append(d)
    out = {}
    for k, ds in per.items():
        if len({d['pred'].shape[0] for d in ds}) == 1:
            out[k] = (np.mean([d['pred'] for d in ds], 0), ds[0]['tgt'], ds[0]['ok'])
    return out


C = {n: load(n) for n in NAMES}
C = {n: v for n, v in C.items() if sorted(v) == [0, 1, 2, 3, 4]}
if not C:
    print('no complete candidates');  raise SystemExit
names = sorted(C)


def score(sel, folds):
    tot = cnt = 0.
    for f in folds:
        p = np.mean([C[c][f][0] for c in sel], 0)
        _, t, ok = C[sel[0]][f]
        e = ((p - t) ** 2)[~ok]
        tot += e.sum();  cnt += len(e)
    return float(np.sqrt(tot / cnt))


def greedy(cands, folds, cap=6):
    sel, cur = [], 1e9
    while len(sel) < cap:
        best = min(((score(sel + [c], folds), c) for c in cands if c not in sel), default=None)
        if best is None or best[0] > cur - 0.02:
            break
        cur, _ = best
        sel.append(best[1])
    return sel


ALL = [0, 1, 2, 3, 4]
sel_all = greedy(names, ALL)
print(f'selected on all five folds: {" ".join(sel_all)}  ->  CV {score(sel_all, ALL):.2f}\n')
print('held-out fold | selected on the OTHER four | CV on all five | the honest number')
hon = []
for f in ALL:
    rest = [x for x in ALL if x != f]
    sel_f = greedy(names, rest)
    s_hon, s_opt = score(sel_f, [f]), score(sel_all, [f])
    hon.append(s_hon)
    print(f'   fold {f}     | {" ".join(sel_f):38s} |  {s_opt:6.2f}    |  {s_hon:6.2f}')
print(f'\npooled held-out RMSE with leave-one-fold-out selection: '
      f'{np.sqrt(np.mean(np.square(hon))):.2f}')
print(f'optimistic number reported in FINDINGS:               {score(sel_all, ALL):.2f}')