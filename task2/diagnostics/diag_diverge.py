"""Why does the encoder-decoder diverge on fold 0?

Fold 0 is the year with the largest level swings, and the enc-dec is the only configuration that
blows up there.  Three candidate causes, tested directly:
  1. expm1 explosion        -- predicted values reach absurd magnitudes
  2. the look-back leaking   -- the encoder sees covariates from the held-out year (a protocol bug)
  3. plain optimisation instability -- it just wanders off

This prints, for each divergent run, the predicted maximum and the number of steps that exceed the
largest value the observed series ever reaches.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

RUNS = os.environ.get('Q2_OUT', 'runs')
y = pd.read_csv('Data/student_train.csv').value.values.astype(np.float64)
ymax = float(y.max())

names = sys.argv[1:] or ['A2_ed168', 'A3_ed_d48', 'B2_ed_clamp', 'base']
print(f'largest value the observed series ever reaches: {ymax:.0f}\n')
for n in names:
    fs = sorted(glob.glob(f'{RUNS}/{n}_f*_s*.npz'))
    if not fs:
        continue
    print(f'--- {n}')
    for f in fs:
        d = np.load(f, allow_pickle=True)
        p, fold, seed = d['pred'], int(d['fold']), int(d['seed'])
        if fold != 0:
            continue
        bad = int((p > ymax).sum())
        print(f'  f{fold} s{seed}: pred max {p.max():10.1f}  mean {p.mean():7.1f}  '
              f'steps > {ymax:.0f}: {bad:5d}  ({100 * bad / len(p):.1f}%)')
    fs2 = sorted(glob.glob(f'{RUNS}/{n}_f4_s*.npz'))
    for f in fs2[:1]:
        d = np.load(f, allow_pickle=True)
        p = d['pred']
        print(f'  (f4 s{int(d["seed"])} for contrast: pred max {p.max():8.1f}  mean {p.mean():6.1f})')