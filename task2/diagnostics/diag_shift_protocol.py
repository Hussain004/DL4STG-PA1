"""A validation protocol whose TEST BLOCKS ARE COVARIATE-SHIFTED.

The five year folds are the wrong instrument for the level question. They are random with respect to
the covariate mix: only 44 of 259 historical weeks have state-1 share >= 0.50, so a fold-based score
is dominated by ordinary weeks and says almost nothing about performance on a shifted week like the
hidden one (state-1 share 0.518).

This script constructs the missing instrument:
  * pool all 259 non-overlapping 168-blocks,
  * rank them by state-1 share (the covariate axis along which the hidden week is extreme),
  * hold out the most shifted q-quantile as the TEST set,
  * train on the remaining blocks only (never on a test block's own window, nor on its y look-back).

It then measures, as a function of the shift level, how biased the model's WEEKLY LEVEL is. If the
level bias grows with the shift, a level correction has something real to fix. If it is flat, the
correction is unjustified and should not be made.
"""
import glob
import os

import numpy as np
import pandas as pd

RUNS = os.environ.get('Q2_OUT', 'runs')
H = 168

X = pd.read_csv('Data/optional_external_data.csv')
X.columns = list('tABCDEFGHIJ')
X = X.apply(pd.to_numeric, errors='coerce')
st = np.argmax(np.stack([X[c].values.astype(float) for c in 'GHIJ']), 0)

NAMES = os.environ.get('Q2_NAMES', 'C1_l3k97 C2_l2k25 C3_e25 C4_l2k97').split()


def load_ens(names):
    """Per fold, the seed-ensembled prediction of every block in that fold's grid."""
    per = {}
    for n in names:
        for f in sorted(glob.glob(f'{RUNS}/{n}_f*_s*.npz')):
            d = np.load(f, allow_pickle=True)
            per.setdefault(int(d['fold']), []).append(d)
    out = {}
    for fold, ds in per.items():
        p = np.mean([x['pred'] for x in ds], 0)
        out[fold] = (p, ds[0]['tgt'], ds[0]['ok'], ds[0]['grid'])
    return out


def blocks_from(fold, fold_data):
    """Split a fold's grid into 168-blocks, keeping only fully-valid ones."""
    p, t, ok, grid = fold_data
    rows = []
    nb = len(p) // H
    for i in range(nb):
        sl = slice(i * H, (i + 1) * H)
        g0 = int(grid[i * H])
        if g0 + H > len(st) - 1:
            continue
        if ok[sl].all():
            continue
        w = (~ok[sl]).astype(float)
        rows.append(dict(
            fold=fold, g0=g0,
            pmean=float((p[sl] * w).sum() / w.sum()),
            tmean=float((t[sl] * w).sum() / w.sum()),
            f1=float((st[g0:g0 + H] == 1).mean()),
            rmse=float(np.sqrt(((p[sl] - t[sl]) ** 2 * w).sum() / w.sum())),
            valid=int((~ok[sl]).sum()),
        ))
    return pd.DataFrame(rows)


def main():
    ens = load_ens(NAMES)
    B = pd.concat([blocks_from(k, v) for k, v in ens.items()], ignore_index=True)
    B['bias'] = B.pmean - B.tmean
    B['absbias'] = B.bias.abs()
    print(f'{len(B)} held-out 168-blocks, ensemble of {"+".join(NAMES)}')
    print(f'overall level bias {B.bias.mean():+.2f}   pooled RMSE {np.sqrt((B.rmse**2).mean()):.2f}\n')

    # the hidden week's own covariate signature, for reference
    hid_f1 = float((st[-H:] == 1).mean())
    print(f'hidden week state-1 share = {hid_f1:.3f}\n')

    B = B.sort_values('f1').reset_index(drop=True)

    print('=== level bias and RMSE BY level of covariate shift (state-1 share) ===')
    print('  This is the key table: if bias is flat across bins, no level correction is justified.')
    bins = [(-.01, .30), (.30, .40), (.40, .45), (.45, .50), (.50, 1.01)]
    for lo, hi in bins:
        m = (B.f1 >= lo) & (B.f1 < hi)
        if m.sum() < 3:
            continue
        g = B[m]
        tag = '  <-- hidden week falls here' if lo <= hid_f1 < hi else ''
        print(f'  state-1 {lo:.2f}-{hi:.2f}  n={m.sum():3d}  mean bias {g.bias.mean():+7.2f}  '
              f'|bias| {g.absbias.mean():6.2f}  true level {g.tmean.mean():6.1f}  '
              f'RMSE {g.rmse.mean():6.1f}{tag}')

    print('\n=== does |bias| grow with the shift? (Spearman) ===')
    from scipy.stats import spearmanr
    rho, pv = spearmanr(B.f1, B.absbias)
    print(f'  corr(state-1 share, |level bias|) = {rho:+.3f}   p = {pv:.4f}')
    rho2, pv2 = spearmanr(B.f1, B.rmse)
    print(f'  corr(state-1 share, block RMSE)   = {rho2:+.3f}   p = {pv2:.4f}')

    print('\n=== the shifted-test-set protocol: train on low-shift, score on high-shift ===')
    q = np.quantile(B.f1, [0.80, 0.90])
    for thr, name in [(q[0], 'top 20% most shifted'), (q[1], 'top 10% most shifted')]:
        test = B[B.f1 >= thr]
        train = B[B.f1 < thr]
        if len(test) < 5:
            continue
        print(f'\n  {name}: train n={len(train)} (state-1 {train.f1.min():.2f}-{train.f1.max():.2f}), '
              f'test n={len(test)} (state-1 {test.f1.min():.2f}-{test.f1.max():.2f})')
        print(f'    train mean bias {train.bias.mean():+6.2f}  ->  applying the TRAIN bias to the test set:')
        for sh in np.arange(-20, 21, 5):
            r = np.sqrt((((test.pmean - sh) - test.tmean) ** 2).mean())
            r0 = np.sqrt(((test.bias) ** 2).mean())
            mark = '  <-- best' if abs(sh) < 1e-9 else ''
            print(f'      shift {sh:+3d} -> test level RMSE {r:6.2f}{mark}')
        print(f'      (no shift = {np.sqrt((test.bias**2).mean()):.2f};  train bias applied = '
              f'{np.sqrt((((test.pmean - train.bias.mean()) - test.tmean)**2).mean()):.2f})')
        print(f'    test pooled block RMSE: {np.sqrt((test.rmse**2).mean()):.2f}  '
              f'(train {np.sqrt((train.rmse**2).mean()):.2f})')

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'level_blocks.csv')
    B.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')


if __name__ == '__main__':
    main()