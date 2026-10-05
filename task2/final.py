"""Train the final ensemble on all five years and write the 168-value submission.

Each member is trained with fold = -1 (the whole observed series) and predicts the hidden block with
rolling origins o = N-168 ... N; every step of the block is then combined over the origins that cover
it with weight (horizon step + 1)^-ow -- exactly what the leave-one-year-out CV measures.  Members
are averaged in physical units.

usage: python final.py OUTNAME SPECFILE
SPECFILE lines:  NAME  EPOCHS  STRIDE  key=value key=value ...     (one line per configuration)
                 a line ending in "SEEDS=0,1,2" overrides the seed list for that line
"""
import shlex
import sys

import numpy as np

from data import H
from train import run, opts_from

SEEDS = [0, 1, 2, 3, 4, 5]


def main():
    args = list(sys.argv[1:])
    outname = 'P2_final'
    spec = None
    for a in args:
        if a.endswith('.txt'):
            spec = a
        else:
            outname = a
    specs = load_spec(spec) if spec else []
    if not specs:
        print('usage: python final.py OUTNAME SPECFILE');  return
    preds, meta = [], []
    for name, epochs, stride, argstr, seeds in specs:
        o = opts_from(shlex.split(argstr))
        for s in seeds:
            nm = f'fin_{name}_s{s}'
            _, P, pavg, _, _ = run(nm, -1, s, epochs, stride, o)
            p = pavg[H:]                 # the grid starts at N-H, so the last H steps are the block
            preds.append(p)
            meta.append((nm, int(P), epochs))
            print(f'[{nm}] mean {p.mean():7.2f} sd {p.std():6.2f} min {p.min():6.2f} '
                  f'max {p.max():7.2f}  P={P}', flush=True)
    Pm = np.stack(preds)
    assert Pm.shape[1] == H, Pm.shape
    fc = np.clip(Pm.mean(0), 0, None)
    assert np.isfinite(fc).all() and (fc >= 0).all(), 'forecast must be finite and non-negative'
    assert len(fc) == 168, len(fc)
    with open(f'{outname}.txt', 'w') as f:
        f.write(','.join(f'{v:.4f}' for v in fc))
    np.save(f'{outname}_members.npy', Pm)
    P, E = sum(m[1] for m in meta), sum(m[2] for m in meta)
    with open(f'{outname}_manifest.txt', 'w') as f:
        f.write('# member\ttrainable parameters\tepochs\n')
        for nm, p, e in meta:
            f.write(f'{nm}\t{p}\t{e}\n')
        f.write(f'TOTAL\t{P}\t{E}\n')
        f.write(f'MEMBERS\t{len(meta)}\n')
        f.write(f'SCORE_PENALTY\t{2e-9*P:.6f}\t{5e-4*E:.6f}\n')
    print(f'\nmembers {len(meta)}   P = {P}   E = {E}   penalty = {2e-9*P + 5e-4*E:.4f}')
    print(f'forecast: mean {fc.mean():.2f} median {np.median(fc):.2f} sd {fc.std():.2f} '
          f'min {fc.min():.2f} max {fc.max():.2f}')
    print('by day:', [round(float(fc[i * 24:(i + 1) * 24].mean()), 1) for i in range(7)])
    print('member means:', np.round(Pm.mean(1), 1))
    print('cross-member sd per step: %.2f' % Pm.std(0).mean())
    print(f'wrote {outname}.txt')


def load_spec(path):
    """NAME EPOCHS STRIDE key=value ... [SEEDS=0,1,2]"""
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        p = shlex.split(line)
        seeds = SEEDS
        rest = []
        for a in p[3:]:
            if a.startswith('SEEDS='):
                seeds = [int(x) for x in a.split('=', 1)[1].split(',')]
            else:
                rest.append(a)
        out.append((p[0], int(p[1]), int(p[2]), ' '.join(rest), seeds))
    return out


if __name__ == '__main__':
    main()