"""Greedy forward selection of ensemble members, on the saved per-step predictions.

Selection is on the pooled 5-fold CV RMSE of the *seed-ensembled* members.  Every candidate must be
a whole configuration (all its seeds averaged), never a single run, because single-run differences
on this series are smaller than the seed spread.
"""
import glob
import os
import sys

import numpy as np

OUT = os.environ.get('Q2_OUT', 'runs')


def rmse_of(preds, tgt, ok):
    e = ((preds - tgt) ** 2)[~ok]
    return float(np.sqrt(e.mean()))


def group(name):
    """Average every seed of one configuration, per fold -> (folds, preds, tgt, ok).

    Configurations trained with a different evs stride have different grid lengths, so folds are
    keyed by fold index and only candidates whose grids all agree can be ensembled together.
    """
    per = {}
    for f in sorted(glob.glob(f'{OUT}/{name}_f*_s*.npz')):
        d = np.load(f, allow_pickle=True)
        per.setdefault(int(d['fold']), []).append(d)
    folds, P, T, O = [], [], [], []
    for k in sorted(per):
        ds = per[k]
        lens = {d['pred'].shape[0] for d in ds}
        if len(lens) != 1:                 # ragged seeds within a fold: skip this candidate
            return [], None, None, None
        folds.append(k)
        P.append(np.mean([d['pred'] for d in ds], 0))
        T.append(ds[0]['tgt'])
        O.append(ds[0]['ok'])
    # every fold is its own array (years differ in length), so only the fold SET must agree
    if sorted(folds) != [0, 1, 2, 3, 4]:
        return [], None, None, None
    return folds, P, T, O


def main(names):
    cands = {}
    for n in names:
        folds, P, T, O = group(n)
        if len(folds) == 5:
            cands[n] = (folds, P, T, O)
    if not cands:
        print('no complete 5-fold candidates');  return

    def score(sel):
        tot = cnt = 0.
        for i in range(5):
            p = np.mean([cands[c][1][i] for c in sel], 0)
            tgt, ok = cands[sel[0]][2][i], cands[sel[0]][3][i]
            e = ((p - tgt) ** 2)[~ok]
            tot += e.sum();  cnt += len(e)
        return float(np.sqrt(tot / cnt))

    solo = sorted(((score([n]), n) for n in cands))
    print('=== single configurations (seed-ensembled, pooled 5-fold) ===')
    for s, n in solo:
        print(f'  {s:6.2f}  {n}')
    sel, cur = [solo[0][1]], solo[0][0]
    print(f'\nstart: {cur:.2f}  [{sel[0]}]')
    while True:
        best = None
        for n in cands:
            if n in sel:
                continue
            s = score(sel + [n])
            if best is None or s < best[0]:
                best = (s, n)
        if best is None or best[0] > cur - 0.02:
            break
        cur, _ = best
        sel.append(best[1])
        print(f'  + {best[1]:24s} -> {cur:.2f}  ({len(sel)} members)')
    print(f'\nSELECTED ({len(sel)}): {" ".join(sel)}')
    print(f'pooled CV RMSE = {cur:.2f}')
    per = [score([n]) for n in sel]
    P = sum(cands[n][0] and 0 for n in sel)
    print(f'mean P per member = '
          f'{sum(int(np.load(sorted(glob.glob(f"{OUT}/{n}_f0_s0.npz"))[0])["params"]) for n in sel) / len(sel):.0f}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    if not args:
        args = sorted({os.path.basename(f).rsplit('_f', 1)[0]
                       for f in glob.glob(f'{OUT}/*_f*_s*.npz')})
    main(args)