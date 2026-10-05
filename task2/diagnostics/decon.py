"""Reverse-engineer the archived configurations from their parameter counts.

The four selected members were produced by an older data.py that had a `ych` flag and used
spec=v0. That code is gone, but every run records its exact trainable parameter count, and the
gsum read-out (Linear(2*n_raw, H)) makes P an almost-linear function of the channel count n_raw
and of n_in = n_raw * (1 + number of trailing windows).  So the archived P values pin down the
configuration far more tightly than guessing.

For AFNet with d=32, H=168, nb bridge features, layers L, dk irrelevant (no params):
    P = 84*d*d + L*(4*d*d + 2*d*mult*d) + 2*d + 2*d*168 + nb*168 + ...
Solving for n_raw and n_in from two archived configurations identifies the channel set.
"""
import glob
import os

import numpy as np

OUT = os.environ.get('Q2_ARCH', os.environ.get('Q2_OLD', 'runs'))
H, d, mult = 168, 32, 2

# what the run files say
print('archived runs (P, epochs):')
for n in ['g3_l3k97', 's3_v1', 's1_e25', 'n1_dk97', 's6_l3', 's2_d25', 's4_v0', 's3_v1']:
    fs = sorted(glob.glob(f'{OUT}/{n}_f*_s*.npz'))
    if not fs:
        continue
    z = np.load(fs[0], allow_pickle=True)
    print(f'  {n:10s} P={int(z["params"]):7d}  epochs={int(z["epochs"]):3d}')


def P_of(n_raw, n_in, layers, nb, gsum_mult=2):
    """Analytic trainable-parameter count for AFNet (verified against the live model).

    gsum_mult: the old harness's gsum was Linear(gsum_mult*n_raw, H) rather than Linear(2*n_raw, H)
    """
    p = 0
    p += n_in * d * 3 + d                  # embed: Conv1d(n_in, d, 3)
    per_block = 4 * (d * d + d)            # AutoCorrelation q,k,v,o
    per_block += (d * mult * d + mult * d) + (mult * d * d + d)   # FFN
    p += layers * per_block
    p += 2 * d                              # LayerNorm
    p += (d + 1)                            # out: Linear(d,1)
    p += (d + 1)                            # tout: Linear(d,1)
    p += gsum_mult * n_raw * H + H         # gsum: Linear(gsum_mult*n_raw, H)
    p += nb * H                             # recent-y bridge gains
    return p


targets = {'g3_l3k97': (45348, 3), 's3_v1': (45379, 2), 's1_e25': (36931, 2), 'n1_dk97': (36931, 2)}
print('\n=== 504 = 3*168 exactly.  Test gsum = Linear(gsum_mult*n_raw, H) ===')
print('(current harness uses 2*n_raw; the archived one appears to use 3*n_raw)')
for gm in (2, 3):
    print(f'\n-- gsum_mult={gm}')
    for name, (Pt, layers) in targets.items():
        hits = []
        for n_raw in range(8, 60):
            for nw in range(1, 6):
                n_in = n_raw * (1 + nw)
                for nb in (7, 10):
                    if abs(P_of(n_raw, n_in, layers, nb, gm) - Pt) <= 4:
                        hits.append((n_raw, nw, nb))
        if hits:
            print(f'   {name:10s} P={Pt} L={layers}: {hits}')


print('\n=== sanity: does the analytic count match the CURRENT harness? ===')
import torch
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from model import AFNet
for spec, layers, nb in [('v1', 2, 10), ('v0', 2, 10), ('v1', 3, 10)]:
    from data import base_channels
    n_raw = len(base_channels(spec))
    n_in = n_raw * 4                       # 1 raw + 3 trailing windows {6,24,168}
    live = sum(q.numel() for q in AFNet(n_in, H=H, d=d, layers=layers, dk=97,
                                        n_raw=n_raw, bridge=(1, 2, 6, 12, 24, 48, 168),
                                        bridge_exact=3).parameters())
    print(f'  spec={spec} layers={layers}: analytic {P_of(n_raw, n_in, layers, nb)}  live {live}  '
          f'{"MATCH" if abs(P_of(n_raw, n_in, layers, nb) - live) <= 2 else "MISMATCH"}')

print('\n=== which (n_raw, n_in) reproduces each archived P? ===')
targets = {'g3_l3k97': (45348, 3), 's3_v1': (45379, 2), 's1_e25': (36931, 2), 'n1_dk97': (36931, 2)}
for name, (Pt, layers) in targets.items():
    hits = []
    for n_raw in range(8, 60):
        for nw in range(1, 6):
            n_in = n_raw * (1 + nw)
            for nb in (7, 10):
                if abs(P_of(n_raw, n_in, layers, nb) - Pt) <= 4:
                    hits.append((n_raw, nw, nb))
    print(f'  {name:10s} P={Pt} layers={layers}: {hits}')

print('\n=== solve for an ADDITIVE extra module of size E on top of a base config ===')
# s4_v0 reproduces exactly at spec=v0, layers=2, nb=10  -> the v0 base is trustworthy
# s1_e25/n1_dk97 = 36931, also layers=2 (dk is parameter-free).
# If the old runs used spec=v0 + ych (adding k channels), then
#   P(v0) + [embed growth from k*(1+nw) inputs] + [gsum growth from k raw] = 36931
for k in range(1, 12):
    for nw in (3, 4):
        # n_raw = 11 + k ; n_in = (11+k)*(1+nw)
        for nb in (7, 10):
            P = P_of(11 + k, (11 + k) * (1 + nw), 2, nb)
            if abs(P - 36931) <= 60:
                print(f'  36931: extra {k} chans -> n_raw={11+k} n_in={(11+k)*(1+nw)} nw={nw} nb={nb} P={P}')
for k in range(1, 12):
    for nw in (3, 4):
        for nb in (7, 10):
            P = P_of(11 + k, (11 + k) * (1 + nw), 3, nb)
            if abs(P - 45348) <= 60:
                print(f'  45348: extra {k} chans -> n_raw={11+k} n_in={(11+k)*(1+nw)} nw={nw} nb={nb} P={P}')