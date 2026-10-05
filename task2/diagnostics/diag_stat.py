"""What statistical preprocessing would the covariates actually benefit from?

Characterises each raw covariate channel for:
  * heavy tails (skew, kurtosis) -> would a rank/quantile transform help?
  * outliers (how much probability mass sits in the top 0.5%) -> winsorising?
  * non-stationarity: variance ratio across the five years, and the lag-1 autocorrelation of the
    DIFFERENCE (a unit root shows up as diff-lag1 near 1) -> differencing / local z-scoring?
  * how much of each channel's variance is the daily and annual cycle -> seasonal differencing?

Everything is measured on the covariate file only; the target is never touched.
"""
import numpy as np
import pandas as pd

X = pd.read_csv('Data/optional_external_data.csv')
X.columns = list('tABCDEFGHIJ')
X = X.apply(pd.to_numeric, errors='coerce').astype(np.float64)
N = len(X)
hod = np.arange(N) % 24
doy = np.arange(N) / 24.0
YRS = [(0, 8760), (8760, 17520), (17520, 26304), (26304, 35064), (35064, 43824)]

print(f'{N} covariate steps\n')
print('channel    mean      sd    skew   kurt   >p99.5   yr-var-ratio  diff-lag1  daily/ann var share')
print('-' * 104)
for c in 'ABCDEFGHIJ':
    v = X[c].values
    sd = v.std()
    z = (v - v.mean()) / (sd + 1e-12)
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    top = float((v > np.quantile(v, 0.995)).mean())

    # variance ratio across the five year blocks: ~1 means a stable scale
    ms = [v[a:b].std() for a, b in YRS]
    ratio = float(max(ms) / (min(ms) + 1e-12))

    # unit-root-ish check: autocorrelation of the first difference
    d = np.diff(v)
    dl1 = float(np.corrcoef(d[:-1], d[1:])[0, 1])

    # variance explained by the daily + annual harmonics (linear projection onto sin/cos)
    cols = [np.ones(N)]
    for p in (24.0, 24.0 * 2):
        cols += [np.sin(2 * np.pi * hod / p), np.cos(2 * np.pi * hod / p)]
    for p in (365.25 * 24.0,):
        cols += [np.sin(2 * np.pi * doy / p), np.cos(2 * np.pi * doy / p)]
    A = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    resid = v - A @ beta
    share = 1 - resid.var() / (v.var() + 1e-12)
    print(f'  {c}  {v.mean():8.2f} {sd:8.2f} {skew:7.2f} {kurt:7.2f} {top:7.3f} '
          f'{ratio:11.2f} {dl1:10.2f} {share:14.3f}')

from scipy.special import ndtri
print('\n=== how much does a RANK (quantile->normal) transform change each channel? ===')
print('  Spearman correlation of the transformed channel with the original, and the tail mass')
for c in 'ABCDEF':
    v = X[c].values
    r = pd.Series(v).rank().to_numpy()
    u = np.clip((r - 0.5) / len(r), 1e-6, 1 - 1e-6)
    g = np.clip(ndtri(u), -6, 6)
    sp = float(pd.Series(v).corr(pd.Series(g), method='spearman'))
    zs = (v - v.mean()) / (v.std() + 1e-12)
    gs = (g - g.mean()) / (g.std() + 1e-12)
    print(f'  {c}: spearman {sp:.3f}   skew {float((zs**3).mean()):+.2f} -> {float((gs**3).mean()):+.2f}')
print('  (probit clamp needed: the extreme tails of E/F are many exact ties, so u hits 0 and 1)')

print('\n=== THE PROBLEM: what the global z-score does to each channel ===')
print('  After standardisation, how many distinct values does a channel take inside +/- 0.5 sigma?')
print('  and how much of its dynamic range is spent on the rare spikes?')
for c in 'ABCDEF':
    v = X[c].values.astype(float)
    z = (v - v.mean()) / (v.std() + 1e-12)
    inner = np.abs(z) < 0.5
    print(f'  {c}: {inner.mean()*100:5.1f}% of steps inside |z|<0.5 ; '
          f'distinct values overall {len(np.unique(v)):5d}, inside |z|<0.5 {len(np.unique(z[inner])):4d} ; '
          f'z range [{z.min():.1f}, {z.max():.1f}]')

print('\n=== E/F are rare spikes: how informative is the SPIKE vs the baseline? ===')
y = pd.read_csv('Data/student_train.csv').value.values.astype(float)
N = len(y)
for c in 'EF':
    v = X[c].values[:N].astype(float)
    on = v > 0
    print(f'  {c}: fires on {on.mean()*100:.2f}% of steps ({on.sum()} of {N})')
    if on.sum() > 10:
        print(f'      y when {c}>0 : mean {y[on].mean():7.2f} median {np.median(y[on]):7.1f} n={on.sum()}')
        print(f'      y when {c}=0 : mean {y[~on].mean():7.2f} median {np.median(y[~on]):7.1f}')
        # the same test on the weekly level, which is what actually matters
        nb = N // 168
        lv = np.array([[y[i*168:(i+1)*168].mean(), (X[c].values[i*168:(i+1)*168] > 0).sum()]
                       for i in range(nb)])
        hi = lv[:, 1] > 0
        if hi.sum() > 3:
            print(f'      weekly level: {c}>0 in {hi.sum()}/{nb} weeks -> mean level '
                  f'{lv[hi,0].mean():.1f} vs {lv[~hi,0].mean():.1f}')

print('\n=== D is a heavy-tailed counter: does log1p suffice? ===')
D = X['D'].values[:N].astype(float)
for name, f in [('raw', lambda z: z), ('log1p', np.log1p), ('sqrt', np.sqrt),
                ('rank-gauss', None)]:
    if f is None:
        continue
    t = f(D)
    zz = (t - t.mean()) / t.std()
    print(f'  {name:9s} skew {float((zz**3).mean()):+6.2f}  kurt {float((zz**4).mean()):7.2f}')

print('\n=== D is a resetting counter: what do its increments look like? ===')
D = X['D'].values
d1 = np.diff(D)
up = d1[d1 > 0]
print(f'  up-steps: n={len(up)}  mean {up.mean():.3f}  max {up.max():.0f}  '
      f'fraction of steps that are a reset {(d1 <= 0).mean():.3f}')
print(f'  D range {D.min():.0f}..{D.max():.0f}; log1p already applied in base_channels')