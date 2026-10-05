"""Why is RMSE near-worst while sMAPE is near-best?

The leaderboard shows this submission with sMAPE 46.07% (2nd best on the board) and RMSE 76.14
(near worst). Those two facts together are informative: relative errors are excellent, but a small
number of steps are badly wrong in ABSOLUTE terms. On a series whose maximum is 994 and whose mean
is 98, that is the signature of missing peaks, not of being systematically miscalibrated.

This measures:
  1. how the squared error is distributed over steps (is it a few steps or everywhere?)
  2. the cost of a single missed peak, in RMSE points
  3. whether the model UNDER-predicts the top of the distribution, and by how much
  4. what the submitted vector's own maximum implies about its peak coverage
"""
import glob
import os

import numpy as np
import pandas as pd

RUNS = os.environ.get('Q2_OUT', 'runs')
H = 168

y = pd.read_csv('Data/student_train.csv').value.values.astype(float)
print(f'target: mean {y.mean():.1f}  median {np.median(y):.1f}  p99 {np.percentile(y,99):.1f}  '
      f'p99.9 {np.percentile(y,99.9):.1f}  max {y.max():.0f}')
print(f'fraction of steps above 300: {np.mean(y>300):.4f}   above 500: {np.mean(y>500):.5f}\n')

sub = np.array([float(x) for x in open('code/P3_final.txt').read().split(',')])
print(f'submitted vector: mean {sub.mean():.1f}  median {np.median(sub):.1f}  max {sub.max():.1f}')
print(f'  -> its maximum is {sub.max():.0f} against a historical maximum of {y.max():.0f}\n')

print('=== 1. how concentrated is the squared error? (pooled over held-out blocks) ===')
per = {}
for n in ['C1_l3k97', 'C2_l2k25', 'C3_e25', 'C4_l2k97']:
    for f in sorted(glob.glob(f'{RUNS}/{n}_f*_s*.npz')):
        d = np.load(f, allow_pickle=True)
        per.setdefault(int(d['fold']), []).append(d)
sq, err_all = [], []
for fold, ds in per.items():
    p = np.mean([x['pred'] for x in ds], 0)
    t, ok = ds[0]['tgt'], ds[0]['ok']
    e = (p - t) ** 2
    sq.append(e[~ok])
    err_all.append(np.abs(p - t)[~ok])
sq = np.concatenate(sq)
err_all = np.concatenate(err_all)
tot = sq.sum()
o = np.argsort(sq)[::-1]
print(f'  pooled RMSE                {np.sqrt(sq.mean()):.2f}')
for frac in (0.001, 0.005, 0.01, 0.05, 0.10):
    k = max(1, int(frac * len(sq)))
    print(f'  top {frac*100:5.1f}% of steps ({k:5d}) carry {sq[o[:k]].sum()/tot*100:5.1f}% of squared error')
rmse_if_top_zeroed = np.sqrt((tot - sq[o[:int(0.01*len(sq))]].sum()) / len(sq))
print(f'  -> RMSE if the worst 1% of steps were perfect: {rmse_if_top_zeroed:.2f}')

print('\n=== 2. what does ONE missed peak cost? ===')
print('  contribution of a single step with error e to the RMSE, over 168 steps:')
for e in (50, 100, 200, 300, 500, 700):
    print(f'    a step off by {e:4d} -> {e/np.sqrt(H):6.2f} RMSE points')
print(f'  a step off by {y.max()-sub.max():.0f} (submitted max vs historical max) -> '
      f'{(y.max()-sub.max())/np.sqrt(H):.2f} RMSE points, if it occurs')

print('\n=== 3. does the model UNDER-predict large y? (regression to the mean check) ===')
ps, ts = [], []
for fold, ds in per.items():
    p = np.mean([x['pred'] for x in ds], 0)
    ps.append(p[~ds[0]['ok']]); ts.append(ds[0]['tgt'][~ds[0]['ok']])
ps = np.concatenate(ps); ts = np.concatenate(ts)
print('   y bin        n      mean y   mean pred   bias')
edges = [0, 25, 50, 100, 200, 300, 500, 2000]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (ts >= lo) & (ts < hi)
    if m.sum() < 20:
        continue
    print(f'  [{lo:4d},{hi:4d})  {m.sum():6d}   {ts[m].mean():7.1f}   {ps[m].mean():7.1f}   '
          f'{ps[m].mean()-ts[m].mean():+7.1f}')
print('\n  -> a NEGATIVE bias in the top bin means the model clips peaks it should have predicted.')

print('\n=== 4. what does the submitted vector look like, hour by hour? ===')
day = [float(sub[i*24:(i+1)*24].mean()) for i in range(7)]
print('  daily means:', np.round(day, 1))
print(f'  within-week sd {sub.std():.1f};  historical within-week sd ~78.5')
print(f'  ratio {sub.std()/78.5:.2f}  -> a flatter forecast than the target typically is')
print('\n  NOTE: for reference, the previous submitted vector had the same shape (corr 0.994).')

print('\n=== 5. IS the shrinkage optimal, or is the model miscalibrated? ===')
print('  Under MSE the conditional MEAN is optimal, so shrinking toward it is correct.')
print('  The real question is whether E[y | prediction] == prediction. If it is, the shrinkage')
print('  is exactly right and nothing is wrong. If E[y|p] > p in the top bins, the model is')
print('  UNDER-confident and a de-shrinkage would genuinely reduce RMSE.')
print('\n   prediction bin        n    mean pred   mean y    E[y|p]-p')
edges = [0, 40, 70, 100, 140, 180, 220, 260, 320, 2000]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (ps >= lo) & (ps < hi)
    if m.sum() < 50:
        continue
    print(f'  [{lo:4d},{hi:4d})  {m.sum():7d}   {ps[m].mean():7.1f}   {ts[m].mean():7.1f}   '
          f'{ts[m].mean()-ps[m].mean():+7.1f}')
print('\n  (positive = truth is higher than predicted = model is UNDER-confident there)')

print('\n=== 6. out-of-fold de-shrinkage test (the decisive one) ===')
print('  Fit E[y|p] on 4 folds by binning the prediction, apply to the 5th, measure pooled RMSE.')
folds = sorted(set(f for f, _ in per.items()))
base_sq, corr_sq, n = 0.0, 0.0, 0
for f in folds:
    tr_folds = [g for g in folds if g != f]
    ptr = np.concatenate([np.mean([x['pred'] for x in per[g]], 0)[~per[g][0]['ok']] for g in tr_folds])
    ttr = np.concatenate([per[g][0]['tgt'][~per[g][0]['ok']] for g in tr_folds])
    pte = np.mean([x['pred'] for x in per[f]], 0)[~per[f][0]['ok']]
    tte = per[f][0]['tgt'][~per[f][0]['ok']]
    qs = np.quantile(ptr, np.linspace(0, 1, 21))
    qs[0] -= 1e-6
    idx = np.clip(np.searchsorted(qs, ptr, 'right') - 1, 0, 19)
    lut = np.array([ttr[idx == k].mean() if (idx == k).sum() > 30 else 0.0 for k in range(20)])
    lut = np.maximum.accumulate(lut)
    b = np.clip(np.searchsorted(qs, pte, 'right') - 1, 0, 19)
    corr = lut[b]
    base_sq += ((pte - tte) ** 2).sum()
    corr_sq += ((corr - tte) ** 2).sum()
    n += len(pte)
print(f'  pooled RMSE, raw model          {np.sqrt(base_sq/n):.2f}')
print(f'  pooled RMSE, out-of-fold mapped  {np.sqrt(corr_sq/n):.2f}')
print(f'  difference: {np.sqrt(base_sq/n) - np.sqrt(corr_sq/n):+.2f}  (positive = de-shrinkage helps)')