"""Where does the within-week shape come from if not from y's own past?

The model must explain shape RMSE ~44 from the covariates alone. This asks:
  1. how much shape is explained by hour-of-day alone (the strongest spectral line),
  2. how much by the 4-way binary state alone,
  3. how much by hour-of-day x state interaction,
and compares each against the pooled RMSE of the model's own residual.
"""
import numpy as np
import pandas as pd

y = pd.read_csv('Data/student_train.csv').value.values.astype(np.float64)
X = pd.read_csv('Data/optional_external_data.csv')
X.columns = list('tABCDEFGHIJ')
X = X.apply(pd.to_numeric, errors='coerce')
N = len(y)
H = 168
G = np.stack([X[c].values.astype(np.float64) for c in 'GHIJ'])
st = np.argmax(G, 0)
hod = np.arange(len(X)) % 24

ok = np.ones(N, bool)
ok[:0] = True

print(f'level: mean {y.mean():.1f}  sd {y.std():.1f}  median {np.median(y):.1f}  max {y.max():.0f}')
print(f'shape sd (within-week, level removed) ~ {y.std():.1f}\n')

# design matrices over the horizon
n = N // H
Y = np.stack([y[i * H:(i + 1) * H] for i in range(n)])
S = np.stack([st[i * H:(i + 1) * H] for i in range(n)])
HO = np.stack([hod[i * H:(i + 1) * H] for i in range(n)])
lvl = Y.mean(1, keepdims=True)


def r2(P):
    return float(np.sqrt(((P - Y) ** 2).mean()))


base = np.full_like(Y, float(y.mean()))
print(f'flat global mean                          RMSE {r2(base):6.2f}')

# 1. hour-of-day fixed effect
fe_h = np.zeros(24)
for h in range(24):
    fe_h[h] = Y[HO == h].mean()
P1 = lvl + fe_h[HO]
print(f'+ hour-of-day fixed effect                RMSE {r2(P1):6.2f}')

# 2. state fixed effect
fe_s = np.zeros(4)
for s in range(4):
    m = S == s
    fe_s[s] = Y[m].mean() if m.any() else 0
P2 = lvl + fe_s[S]
print(f'+ 4-way state fixed effect                RMSE {r2(P2):6.2f}')

# 3. hour-of-day x state
fe_hs = np.zeros((4, 24))
for s in range(4):
    for h in range(24):
        m = (S == s) & (HO == h)
        fe_hs[s, h] = Y[m].mean() if m.sum() > 20 else fe_h[h]
P3 = lvl + fe_hs[S, HO]
print(f'+ hour-of-day x state interaction         RMSE {r2(P3):6.2f}')

print('\n=== how big is the level part vs the shape part? ===')
r = Y - P3
lv_err = (r.mean(1))
sh_err = r - lv_err[:, None]
print(f'  with the hour x state profile: level RMSE {np.sqrt((lv_err**2).mean()):6.2f}   '
      f'shape RMSE {np.sqrt((sh_err**2).mean()):6.2f}')
print(f'  -> shape accounts for '
      f'{(sh_err**2).mean() / ((r**2).mean()):.0%} of the remaining squared error')

print('\n=== state-1 share vs weekly level (the level-shift question, non-parametric) ===')
sh1 = (S == 1).mean(1)
for lo, hi in zip(np.arange(0, 1.0, 0.1), np.arange(0.1, 1.01, 0.1)):
    m = (sh1 >= lo) & (sh1 < hi)
    if m.sum() >= 5:
        print(f'  state-1 share {lo:.1f}-{hi:.1f}  n={m.sum():3d}  mean weekly level {Y[m].mean():6.1f}  '
              f'median {np.median(Y[m]):6.1f}')