"""Encoder-only Autoformer for the 168-step block.

The optional file is known over the hidden week, so this is a *nowcaster*: it reads the covariate
channels of the 168 steps it must predict and emits one value per step (no decoder, no recurrence).

One block = Auto-Correlation -> Series Decomposition -> FFN -> Series Decomposition, with residual
connections.  Both are the Autoformer mechanisms of Wu et al. (2021):

  * AutoCorrelation mixes by *time delay* instead of by position pair (blocks.AutoCorrelation).
  * SeriesDecomp splits every residual stream into a moving-average trend and a seasonal remainder;
    the seasonal part goes on, and every trend that was removed is summed up and added to the
    output, so the slow structure is never thrown away.

Three read-outs, all zero-initialised so that training starts exactly at the plain model:

  seasonal   per-step linear map of the LayerNormed seasonal stream.
  trend      per-step linear map of the accumulated trends (the decomposition's slow part).
  window     per-step linear map of two summaries of the *raw* covariates: their mean over the 168
             steps being predicted, and their trailing mean at the forecast origin.  This hands the
             read-out the level of the predicted window directly instead of making two position-wise
             layers re-derive it.
  recent y   per-horizon-step gains on a handful of summaries of the y values just before the origin
             (trailing means and the exact last lags).  y never enters the encoder as a sequence, so
             the network sees a dozen scalars of memory and cannot memorise a training window.

Output: expm1 of a standardised log-scale, trained with MSE in physical units (the leaderboard metric).
"""
import torch
import torch.nn as nn

from autoformer_blocks import SeriesDecomp, AutoCorrelation, FFN


class Block(nn.Module):
    """decomp: 1 = decompose after the mixer's residual only, 2 = after both sub-layers, 0 = off."""

    def __init__(self, d, heads, drop, mixer, decomp, dk):
        super().__init__()
        self.mixer = mixer
        self.mix = AutoCorrelation(d, heads, drop) if mixer == 'autocorr' \
            else nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn = FFN(d, 2, drop)
        self.drop = nn.Dropout(drop)
        self.d1 = SeriesDecomp(dk) if decomp in (1, 2) else None
        self.d2 = SeriesDecomp(dk) if decomp == 2 else None

    def forward(self, x):
        m = self.mix(x) if self.mixer == 'autocorr' else self.mix(x, x, x, need_weights=False)[0]
        x = x + self.drop(m)
        trend = 0
        if self.d1 is not None:
            x, t = self.d1(x)
            trend = trend + t
        x = x + self.drop(self.ffn(x))
        if self.d2 is not None:
            x, t = self.d2(x)
            trend = trend + t
        return x, trend


class WindowReadout(nn.Module):
    """Per-step read-out of the covariate summary of the window being predicted.

    gmlp = 0  -> a single linear map (2*n_raw -> H): the level is a linear function of the
                  covariate means, which cannot express a threshold such as "weeks whose state-1
                  share is above 0.5 sit at a much lower level".
    gmlp > 0  -> the same input through a 2-layer MLP with a GELU, so the weekly level may be a
                  non-linear function of the covariates.  The final layer is zero-initialised, so
                  the model starts exactly at the linear read-out.
    """

    def __init__(self, n_raw, H, gmlp=0):
        super().__init__()
        self.n_raw = n_raw
        if gmlp:
            self.net = nn.Sequential(nn.Linear(2 * n_raw, gmlp), nn.GELU(), nn.Linear(gmlp, H))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.net = nn.Linear(2 * n_raw, H)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)

    def forward(self, x):
        w = x[:, :, :self.n_raw]                                   # raw channels of the window
        return self.net(torch.cat([w.mean(1), x[:, 0, :self.n_raw]], 1))


class AFNet(nn.Module):
    def __init__(self, n_in, H=168, d=32, heads=4, layers=2, dk=97, drop=0.1,
                 mixer='autocorr', decomp=1, n_raw=0, bridge=(), bridge_exact=0, gmlp=0):
        super().__init__()
        self.H, self.n_raw = H, int(n_raw)
        self.embed = nn.Conv1d(n_in, d, 3, padding=1, padding_mode='replicate')
        self.blocks = nn.ModuleList([Block(d, heads, drop, mixer, decomp, dk) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.tout = nn.Linear(d, 1) if decomp else None          # read-out of the accumulated trends
        self.gsum = WindowReadout(self.n_raw, H, gmlp) if self.n_raw else None
        nb = len(bridge) + int(bridge_exact)
        self.g = nn.Parameter(torch.zeros(nb, H)) if nb else None
        for m in (self.out, self.tout):
            if m is not None:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, xb=None, yb=None):
        """x: [B, H, n_in] covariates of the steps to predict.  yb: [B, n_bridge] recent-y summaries.
        xb is accepted and ignored: the encoder-only model has no look-back stream."""
        if yb is None:
            yb = xb
        h = self.embed(x.transpose(1, 2)).transpose(1, 2)            # [B, H, d]
        trend = 0
        for b in self.blocks:
            h, t = b(h)
            trend = trend + t
        z = self.out(self.norm(h))[..., 0]                         # [B, H]
        if self.tout is not None:
            z = z + self.tout(trend)[..., 0]
        if self.gsum is not None:
            z = z + self.gsum(x)
        if self.g is not None and yb is not None:
            z = z + yb @ self.g
        return z                                                    # [B, H]


def _selftest():
    torch.manual_seed(0)
    for mixer in ('autocorr', 'attention'):
        for decomp in (0, 1, 2):
            m = AFNet(20, mixer=mixer, decomp=decomp)
            assert m(torch.randn(3, 168, 20)).shape == (3, 168), (mixer, decomp)
    m = AFNet(20, n_raw=20, bridge=(1, 6, 24), bridge_exact=1)
    assert m(torch.randn(3, 168, 20), torch.randn(3, 4)).shape == (3, 168)
    # the removed trend must reach the output
    m = AFNet(5, decomp=1).eval()
    x = torch.randn(2, 168, 5)
    for p in m.parameters():
        nn.init.normal_(p, 0, 0.1)
    with torch.no_grad():
        a = m(x)
        m.tout.weight.zero_()
        m.tout.bias.zero_()
        assert (a - m(x)).abs().max() > 1e-6
    # a change to the LAST input step must be able to change the output
    m = AFNet(5).eval()
    for p in m.parameters():
        nn.init.normal_(p, 0, 0.1)
    with torch.no_grad():
        x2 = x.clone()
        x2[:, -1] += 1.
        assert (m(x) - m(x2)).abs().max() > 1e-6
    print(f'model selftest ok; params = {sum(p.numel() for p in AFNet(44, n_raw=11, bridge=(1, 6, 24, 168), bridge_exact=3).parameters())}')


if __name__ == '__main__':
    _selftest()
