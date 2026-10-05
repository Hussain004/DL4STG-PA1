"""Summarise finished runs: per-run RMSE, seed ensembles, pooled RMSE, and (optionally) the
ensemble of several configurations.  All numbers come from the saved per-step predictions.

usage: python summ.py NAME [NAME ...]        # one line block per name, plus a combined block
       python summ.py --csv                   # machine-readable table of every run on disk
"""
import sys
import glob
import os
import numpy as np

OUT = os.environ.get('Q2_OUT', 'runs')


def load(names):
    runs = {}
    for n in names:
        for f in sorted(glob.glob(f'{OUT}/{n}_f*_s*.npz')):
            d = np.load(f, allow_pickle=True)
            runs[(int(d['fold']), int(d['seed']))] = d
    return runs


def rmse(p, t, ok):
    return float(np.sqrt((((p - t) ** 2) * ~ok).sum() / (~ok).sum()))


def pooled(pairs):
    """pairs: list of (pred, tgt, ok). Returns the RMSE over all pooled valid steps."""
    tot = cnt = 0.
    for p, t, ok in pairs:
        e = ((p - t) ** 2)[~ok]
        tot += e.sum()
        cnt += len(e)
    return float(np.sqrt(tot / cnt))


def summarise(names, ens=True, quiet=False, tag=None):
    runs = load(names)
    if not runs:
        return None
    folds = sorted({f for f, _ in runs})
    seeds = sorted({s for _, s in runs})
    single = [rmse(d['pred'], d['tgt'], d['ok']) for d in runs.values()]
    per_seed, per_fold, ens_pairs, single_pairs = {}, {}, [], []
    for s in seeds:
        ps = [rmse(runs[(f, s)]['pred'], runs[(f, s)]['tgt'], runs[(f, s)]['ok'])
              for f in folds if (f, s) in runs]
        if ps:
            per_seed[s] = pooled([(runs[(f, s)]['pred'], runs[(f, s)]['tgt'], runs[(f, s)]['ok'])
                                  for f in folds if (f, s) in runs])
    for f in folds:
        ss = [s for s in seeds if (f, s) in runs]
        if not ss:
            continue
        per_fold[f] = rmse(np.mean([runs[(f, s)]['pred'] for s in ss], 0), runs[(f, ss[0])]['tgt'],
                           runs[(f, ss[0])]['ok'])
        ens_pairs.append((np.mean([runs[(f, s)]['pred'] for s in ss], 0), runs[(f, ss[0])]['tgt'],
                          runs[(f, ss[0])]['ok']))
        for s in ss:
            single_pairs.append((runs[(f, s)]['pred'], runs[(f, s)]['tgt'], runs[(f, s)]['ok']))
    out = dict(tag=tag or '+'.join(names), n_runs=len(single),
               single_pooled=pooled(single_pairs),
               seedens_pooled=pooled(ens_pairs) if ens_pairs else float('nan'),
               folds={f: round(per_fold[f], 2) for f in per_fold},
               seeds={s: round(per_seed[s], 2) for s in per_seed},
               params=int(next(iter(runs.values()))['params']),
               C=-1)
    # ensemble over every run on disk with these names
    allp = {}
    for (f, s), d in runs.items():
        allp.setdefault(f, []).append(d['pred'])
    ap = []
    for f, ps in allp.items():
        d0 = runs[(f, sorted(s for (ff, s) in runs if ff == f)[0])]
        ap.append((np.mean(ps, 0), d0['tgt'], d0['ok']))
    out['allens_pooled'] = pooled(ap) if len(names) > 1 else out['seedens_pooled']
    if not quiet:
        print(f'{out["tag"]:26s} runs={out["n_runs"]:3d} P={out["params"]:7d} C={out["C"]:4d} | '
              f'single {out["single_pooled"]:6.2f} | seed-ens {out["seedens_pooled"]:6.2f} | '
              f'all-ens {out["allens_pooled"]:6.2f} | folds ' +
              ' '.join(f'{out["folds"][f]:.1f}' for f in sorted(out['folds'])))
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    rows = []
    for a in args:
        r = summarise([a], ens=True, tag=a)
        if r:
            rows.append(r)
    if len(args) > 1:
        rows.append(summarise(args, ens=True, tag='ALL(' + '+'.join(args) + ')'))
    if '--csv' in sys.argv:
        for r in rows:
            print(r['tag'], r['single_pooled'], r['seedens_pooled'], r['allens_pooled'], r['folds'], sep='\t')
