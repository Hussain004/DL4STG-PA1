"""Is the within-week SHAPE of y predictable from the PREVIOUS week's shape?

The submitted model sees y only as ~7 scalars (trailing means + 3 exact lags), so it cannot know
last week's hourly profile. If shape[k] ~ shape[k-168] is strong, that is a large amount of free
information the model is currently throwing away.

Reported: shape correlation at several lags, an oracle "copy last week's profile" forecaster, and
a shrinkage blend of the two. All on non-overlapping 168-blocks, years 0-4.
"""
import numpy as np
import pandas as pd

y = pd.read_csv('Data/student_train.csv').value.values.astype(np.float64)
N = len(y)
H = 168

# interpolate-mask, same rule as data.py
d1 = np.diff(y)


def _runs(mask, minlen):
    out = np.zeros(len(mask), bool)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= minlen:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


BAD = np.zeros(N, bool)
c = _runs(d1 == 0, 3)
BAD[:-1] |= c
BAD[1:] |= c
l = _runs((np.abs(np.diff(y, 2)) < 1e-9) & (d1[1:] != 0), 2)
BAD[:-2] |= l
BAD[1:-1] |= l
BAD[2:] |= l

# non-overlapping blocks
starts = np.arange(0, N - H + 1, H)
B = np.stack([y[s:s + H] for s in starts])          # [nb, H]
M = ~np.stack([BAD[s:s + H] for s in starts])      # valid mask
nb = len(B)
print(f'{nb} non-overlapping 168-blocks, mean level {B[M].mean():.1f}\n')

lv = B.sum(1) / M.sum(1)                            # per-block level
sh = (B - lv[:, None]) * M                          # shape, level removed


def corr(a, b, m):
    if m.ndim == 2:
        m = m.all(1)
    a, b, m = np.asarray(a)[m], np.asarray(b)[m], m
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


print('=== shape correlation, block i vs block i-j ===')
for j in (1, 2, 3, 4, 13, 26, 52):
    if j < nb:
        m = M[j:] & M[:-j]
        print(f'  lag {j:3d} blocks ({j*7:4d} d):  shape corr {corr(sh[j:], sh[:-j], m):+.3f}   '
              f'level corr {corr(lv[j:], lv[:-j], M[j:] & M[:-j]):+.3f}')

print('\n=== how good is each one-piece weekly forecaster? (pooled RMSE over all blocks) ===')
mm = np.ones((nb, H), bool)                          # blocks with no bad steps


def rep(name, P):
    r = float(np.sqrt((((P - B) ** 2) * mm).sum() / mm.sum()))
    print(f'  {name:44s} {r:6.2f}')
    return r


rep('global mean', np.full_like(B, float(y[~BAD].mean())))
rep('copy previous week level', np.stack([np.full(H, lv[i - 1]) for i in range(nb)]))
rep('copy previous week FULL profile', B[np.maximum(np.arange(nb) - 1, 0)])
rep('copy prev week profile, level from prev', sh[np.maximum(np.arange(nb) - 1, 0)] + lv[:, None])

print('\n=== blend: level from previous week, shape shrunk toward the average profile ===')
prev = np.maximum(np.arange(nb) - 1, 0)
avg_shape = sh.mean(0)
best = None
for a in np.arange(0, 1.01, 0.05):
    P = lv[:, None] + a * (sh[prev] - avg_shape) + (1 - a) * avg_shape
    r = float(np.sqrt((((P - B) ** 2) * mm).sum() / mm.sum()))
    if best is None or r < best[1]:
        best = (a, r)
print(f'  best shrinkage a = {best[0]:.2f}  ->  RMSE {best[1]:.2f}')

print('\n=== how much of the shape is a stable weekly profile? ===')
print(f'  sd of shape across blocks, averaged over the 168 positions: {sh.std(0).mean():.2f}')
print(f'  sd of the AVERAGE shape (a stable profile would be ~0):    {avg_shape.std():.2f}')
print(f'  -> repeatable share ~ 1 - {avg_shape.std()**2 / sh.std(0).mean()**2:.3f}')
for k in (0, 6, 12, 18, 24, 36, 48, 72, 96, 120, 144):
    print(f'    hour-of-block {k:3d} (h {k % 24:02d})  mean shape {avg_shape[k]:+7.2f}  '
          f'across-block sd {sh[:, k].std():6.2f}')