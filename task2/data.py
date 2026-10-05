"""Data loading, covariate channels and the fold protocol for Task 2 (the 168-step block).

1 step = 1 hour: the file splits into year-blocks of exactly 8760/8784 hours, so the 168-step
horizon is one week.  Every channel below is a function of the optional file at times <= the time
being predicted, so no target information enters the inputs.
"""
import os

import numpy as np
import pandas as pd

DATA = os.environ.get('Q2_DATA', 'Data/')
H = 168

y_all = pd.read_csv(DATA + 'student_train.csv').value.values.astype(np.float64)
X = pd.read_csv(DATA + 'optional_external_data.csv')
X.columns = list('tABCDEFGHIJ')
X = X.apply(pd.to_numeric, errors='coerce')
N, NT = len(y_all), len(X)                                   # 43,656 observed, 43,824 in total

CYCLES = [(0, 8760), (8760, 17520), (17520, 26304), (26304, 35064), (35064, 43824)]


def _runs(mask, minlen):
    """True on the interior of every run of True of length >= minlen."""
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


def interpolated_mask():
    """Steps that look filled in: 4+ identical values in a row, or 4+ on a perfectly straight line.
    The handout says the hidden block has none, so they are left out of the loss and of scoring."""
    y, d1 = y_all, np.diff(y_all)
    m = np.zeros(N, bool)
    c = _runs(d1 == 0, 3)
    m[:-1] |= c
    m[1:] |= c
    l = _runs((np.abs(np.diff(y, 2)) < 1e-9) & (d1[1:] != 0), 2)
    m[:-2] |= l
    m[1:-1] |= l
    m[2:] |= l
    return m


BAD = interpolated_mask()

_R = {c: X[c].values.astype(np.float64) for c in 'ABCDEFGHIJ'}
_A, _B, _C, _D, _E, _F, _G, _H, _I, _J = (_R[c] for c in 'ABCDEFGHIJ')
_drop = np.r_[True, np.diff(_D) < 0]                        # D is a running total that resets
_rate = np.where(_drop, _D, np.r_[_D[0], np.diff(_D)])      # its per-step increment
_state = np.argmax(np.stack([_G, _H, _I, _J]), 0).astype(np.float64)   # the 4 indicators -> one id
_hod = (np.arange(NT) % 24).astype(np.float64)             # hour of day
_doy = np.arange(NT) / 24.0                                # day of year


def _rankgauss(v):
    """Rank-transform to an approximate standard normal (a quantile / Gaussianisation transform).

    Motivation (diag/diag_stat.py): A, B and C carry only 60-69 distinct values, and E carries 28 with
    99.2% of steps exactly zero.  After a plain z-score, E spans z in [-0.1, +35], so 99% of steps
    read as indistinguishable from the baseline and the rare spikes dominate the scale.  Ranking
    spends the same dynamic range evenly over the observed quantiles instead of over the outliers.
    Ties share the average rank, so a channel that is zero 99% of the time becomes a clean two-level
    variable (about -3.1 when zero, up to +3.1 when firing) rather than a spike on a huge scale.
    """
    r = pd.Series(v).rank(method='average').to_numpy()
    u = np.clip((r - 0.5) / len(r), 1e-6, 1 - 1e-6)
    return np.clip(_ndtri(u), -6.0, 6.0)


try:
    from scipy.special import ndtri as _ndtri          # inverse standard-normal CDF
except ImportError:                                     # Acklam's rational approximation
    def _ndtri(p):
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        p = np.asarray(p, dtype=np.float64)
        lo, hi = 0.02425, 1 - 0.02425
        out = np.empty_like(p)

        def tail(q, sgn):
            q = np.sqrt(-2 * np.log(q))
            return sgn * (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                         ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        m = p < lo
        out[m] = tail(p[m], 1.0)
        m = p > hi
        out[m] = tail(1 - p[m], -1.0)
        m = ~(m | (p < lo))
        q = p[m] - 0.5
        r = q * q
        out[m] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                 (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        return out


def base_channels(spec='v1'):
    """The covariate channels.  v0 = the ten raw variables; v1 adds a few obviously motivated
    transforms -- the level-polynomials of B, |A|, and the daily and annual harmonics, which is how
    the model gets at the 24 h cycle that y's spectrum is dominated by.

    spec='g' applies the statistical preprocessing measured in diag/diag_stat.py: a rank-gauss
    transform on the coarsely-quantised / spiky channels (A, B, C, E, F) and an explicit
    spike indicator for the rare-spike counters E and F, whose fires are strongly associated with a
    higher level (E>0: mean y 136.6 against 97.9; weeks containing E average 122.8 against 94.8).
    D is deliberately left alone -- log1p already brings it from skew 4.31/kurt 26.6 to 0.70/2.60.
    """
    if spec == 'g':
        ch = {
            'A': _rankgauss(_A), 'B': _rankgauss(_B), 'C': _rankgauss(_C),
            'logD': np.log1p(_D) / 2., 'logw': np.log1p(np.maximum(_rate, 0)) / 2.,
            'E': _rankgauss(_E), 'F': _rankgauss(_F),
            'Eon': (_E > 0).astype(np.float64), 'Fon': (_F > 0).astype(np.float64),
            'G': _G, 'H': _H, 'I': _I, 'J': _J,
        }
        ch['st'] = _state
        ch['Bp2'] = (_B / 10.) ** 2 / 10.
        ch['Bp3'] = (_B / 10.) ** 3 / 100.
        ch['An'] = np.abs(_A) / 10.
        ch['tod_s'] = np.sin(2 * np.pi * _hod / 24)
        ch['tod_c'] = np.cos(2 * np.pi * _hod / 24)
        ch['doy_s'] = np.sin(2 * np.pi * _doy / 365.25)
        ch['doy_c'] = np.cos(2 * np.pi * _doy / 365.25)
        return ch
    if spec == 'gr':                                   # rank-gauss only, no extra spike channels
        return {'A': _rankgauss(_A), 'B': _rankgauss(_B), 'C': _rankgauss(_C),
                'logD': np.log1p(_D) / 2., 'logw': np.log1p(np.maximum(_rate, 0)) / 2.,
                'E': _rankgauss(_E), 'F': _rankgauss(_F),
                'G': _G, 'H': _H, 'I': _I, 'J': _J}
    if spec == 'gw':
        # Round G showed a GLOBAL rank transform hurts: it makes every E spike nearly the same value
        # ("this step is in the top 0.8% of five years") and throws away how BIG the spike was, which
        # is the informative part.  This variant keeps the raw magnitude but makes it scale-free and
        # explicit: log1p of the spike size, and the log of the spike COUNT in each trailing window.
        ch = {
            'A': _A / 10., 'B': _B / 10., 'C': (_C - 1016) / 10.,
            'logD': np.log1p(_D) / 2., 'logw': np.log1p(np.maximum(_rate, 0)) / 2.,
            'E': np.log1p(_E) / 2., 'F': np.log1p(_F) / 2.,
            'Eon': (_E > 0).astype(np.float64), 'Fon': (_F > 0).astype(np.float64),
            'G': _G, 'H': _H, 'I': _I, 'J': _J,
        }
        ch['st'] = _state
        ch['Bp2'] = (_B / 10.) ** 2 / 10.
        ch['Bp3'] = (_B / 10.) ** 3 / 100.
        ch['An'] = np.abs(_A) / 10.
        ch['tod_s'] = np.sin(2 * np.pi * _hod / 24)
        ch['tod_c'] = np.cos(2 * np.pi * _hod / 24)
        ch['doy_s'] = np.sin(2 * np.pi * _doy / 365.25)
        ch['doy_c'] = np.cos(2 * np.pi * _doy / 365.25)
        return ch
    if spec == 'gi':                    # spike indicators only, NO rank transform (isolates which half helps)
        ch = {
            'A': _A / 10., 'B': _B / 10., 'C': (_C - 1016) / 10.,
            'logD': np.log1p(_D) / 2., 'logw': np.log1p(np.maximum(_rate, 0)) / 2.,
            'E': _E / 5., 'F': _F / 5., 'G': _G, 'H': _H, 'I': _I, 'J': _J,
            'Eon': (_E > 0).astype(np.float64), 'Fon': (_F > 0).astype(np.float64),
        }
    else:
        ch = {
            'A': _A / 10., 'B': _B / 10., 'C': (_C - 1016) / 10.,
            'logD': np.log1p(_D) / 2., 'logw': np.log1p(np.maximum(_rate, 0)) / 2.,
            'E': _E / 5., 'F': _F / 5., 'G': _G, 'H': _H, 'I': _I, 'J': _J,
        }
    if spec == 'v0':
        return ch
    ch['st'] = _state
    ch['Bp2'] = (_B / 10.) ** 2 / 10.
    ch['Bp3'] = (_B / 10.) ** 3 / 100.
    ch['An'] = np.abs(_A) / 10.
    ch['tod_s'] = np.sin(2 * np.pi * _hod / 24)
    ch['tod_c'] = np.cos(2 * np.pi * _hod / 24)
    ch['doy_s'] = np.sin(2 * np.pi * _doy / 365.25)
    ch['doy_c'] = np.cos(2 * np.pi * _doy / 365.25)
    return ch


def covariates(spec='v1', wins=(6, 24, 168)):
    """Raw channels plus causal trailing means over [t-w+1, t] for each of them.  Trailing, not
    centred: a window that ends at t uses nothing after t."""
    ch = base_channels(spec)
    names = list(ch)
    Z = np.stack([ch[n] for n in names], 1)
    t = np.arange(NT)
    cs = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z, 0)])
    extra, wnames = [], []
    for w in wins:
        lo = np.maximum(t - w + 1, 0)
        extra.append((cs[t + 1] - cs[lo]) / (t + 1 - lo)[:, None])
        wnames += [f'{n}_w{w}' for n in names]
    if extra:
        names = names + wnames
        Z = np.hstack([Z] + extra)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    return np.ascontiguousarray(Z, dtype=np.float32), names
