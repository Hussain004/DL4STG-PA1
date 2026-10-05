"""Encoder-decoder Autoformer for the 168-step block.

EncoderBlock : autocorr -> series decomp -> FFN, trends accumulated.
DecoderBlock : self-mix -> decomp -> cross-attention -> decomp -> FFN -> decomp.

The decoder's cross-attention is plain point-wise MultiheadAttention on purpose: Auto-Correlation
discovers periodicity *within one series*, whereas reading a different sequence (the look-back) is
exactly the job point-wise attention is good at.  Both Autoformer mechanisms are present and
ablatable in both stacks.
"""
import torch
import torch.nn as nn

from autoformer_blocks import SeriesDecomp, AutoCorrelation, FFN
from model import WindowReadout


def _mixer(kind, d, heads, drop):
    return AutoCorrelation(d, heads, drop) if kind == 'autocorr' \
        else nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)


class EncoderBlock(nn.Module):
    def __init__(self, d, heads, drop, mixer, decomp, dk):
        super().__init__()
        self.mixer = mixer
        self.mix = _mixer(mixer, d, heads, drop)
        self.ffn = FFN(d, 2, drop)
        self.drop = nn.Dropout(drop)
        self.d1 = SeriesDecomp(dk) if decomp else None

    def forward(self, x):
        m = self.mix(x) if self.mixer == 'autocorr' else self.mix(x, x, x, need_weights=False)[0]
        x = x + self.drop(m)
        trend = 0
        if self.d1 is not None:
            x, t = self.d1(x)
            trend = trend + t
        return x + self.drop(self.ffn(x)), trend


class DecoderBlock(nn.Module):
    def __init__(self, d, heads, drop, mixer, decomp, dk):
        super().__init__()
        self.mixer = mixer
        self.mix = _mixer(mixer, d, heads, drop)
        self.cross = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ffn = FFN(d, 2, drop)
        self.drop = nn.Dropout(drop)
        self.d1 = SeriesDecomp(dk) if decomp else None
        self.d2 = SeriesDecomp(dk) if decomp else None

    def forward(self, x, mem):
        m = self.mix(x) if self.mixer == 'autocorr' else self.mix(x, x, x, need_weights=False)[0]
        x = x + self.drop(m)
        trend = 0
        if self.d1 is not None:
            x, t = self.d1(x)
            trend = trend + t
        x = x + self.drop(self.cross(x, mem, mem, need_weights=False)[0])
        if self.d2 is not None:
            x, t = self.d2(x)
            trend = trend + t
        return x + self.drop(self.ffn(x)), trend
class AFEncDec(nn.Module):
    """Encoder-decoder Autoformer.

    Encoder  : a trailing look-back of L steps of the same covariate channels.
    Decoder  : the H steps to predict, cross-attending to the encoder memory.

    Read-outs, all zero-initialised so training starts exactly at the plain model:
      seasonal  per-step linear map of the LayerNormed decoder stream.
      trend     per-step linear map of the accumulated decoder trends.
      etrend    per-step linear map of the accumulated encoder trends (the look-back's slow part).
      window    per-step linear map of the covariate means over the 168 predicted steps plus the
                trailing mean at the origin -- hands the read-out the weekly level directly.
      recent y  per-horizon-step gains on a handful of y summaries from before the origin.
    """

    def __init__(self, n_in, H=168, L=168, d=32, heads=4, elayers=2, dlayers=1,
                 dk=97, drop=0.1, mixer='autocorr', decomp=1, n_raw=0, bridge=(), bridge_exact=0,
                 gmlp=0):
        super().__init__()
        self.H, self.L, self.n_raw = H, int(L), int(n_raw)
        self.embed = nn.Conv1d(n_in, d, 3, padding=1, padding_mode='replicate')
        self.dembed = nn.Conv1d(n_in, d, 3, padding=1, padding_mode='replicate')
        self.eblocks = nn.ModuleList([EncoderBlock(d, heads, drop, mixer, decomp, dk)
                                      for _ in range(elayers)])
        self.dblocks = nn.ModuleList([DecoderBlock(d, heads, drop, mixer, decomp, dk)
                                      for _ in range(dlayers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.tout = nn.Linear(d, 1) if decomp else None
        self.etout = nn.Linear(d, 1) if decomp else None
        self.gsum = WindowReadout(self.n_raw, H, gmlp) if self.n_raw else None
        nb = len(bridge) + int(bridge_exact)
        self.g = nn.Parameter(torch.zeros(nb, H)) if nb else None
        for m in (self.out, self.tout, self.etout):
            if m is not None:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, xb=None, yb=None):
        """x: [B,H,n_in] covariates of the steps to predict.
        xb: [B,L,n_in] covariates of the L steps before the origin.  yb: [B,n_bridge] recent y."""
        dt = self.dembed(x.transpose(1, 2)).transpose(1, 2)
        trend, etrend, mem = 0, 0, None
        if xb is not None:
            eh = self.embed(xb.transpose(1, 2)).transpose(1, 2)
            for b in self.eblocks:
                eh, t = b(eh)
                etrend = etrend + t
            mem = eh
        for b in self.dblocks:
            dt, t = b(dt, mem) if mem is not None else b(dt, dt)
            trend = trend + t
        z = self.out(self.norm(dt))[..., 0]
        if self.tout is not None and torch.is_tensor(trend):
            z = z + self.tout(trend)[..., 0]
        if self.etout is not None and torch.is_tensor(etrend):
            z = z + self.etout(etrend)[..., 0]
        if self.gsum is not None:
            z = z + self.gsum(x)
        if self.g is not None and yb is not None:
            z = z + yb @ self.g
        return z


def _selftest():
    torch.manual_seed(0)
    for mixer in ('autocorr', 'attention'):
        for decomp in (0, 1):
            for el, dl in ((2, 1), (0, 1)):
                m = AFEncDec(20, elayers=el, dlayers=dl, mixer=mixer, decomp=decomp)
                assert m(torch.randn(3, 168, 20), torch.randn(3, 168, 20)).shape == (3, 168)
    m = AFEncDec(20, n_raw=20, bridge=(1, 6, 24), bridge_exact=1).eval()
    for p in m.parameters():
        nn.init.normal_(p, 0, 0.1)
    x, xb, yb = torch.randn(2, 168, 20), torch.randn(2, 168, 20), torch.randn(2, 4)
    with torch.no_grad():
        a = m(x, xb, yb)
        m.tout.weight.zero_(); m.tout.bias.zero_()
        assert (a - m(x, xb, yb)).abs().max() > 1e-6, 'decoder trend must reach the output'
        m.tout.weight.zero_(); m.tout.bias.zero_()
        m.etout.weight.zero_(); m.etout.bias.zero_()
        assert (a - m(x, xb, yb)).abs().max() > 1e-6, 'encoder trend must reach the output'
        # no look-back at all must still work (elayers=0 path)
        m0 = AFEncDec(20, elayers=0, dlayers=1, decomp=1).eval()
        for p in m0.parameters():
            nn.init.normal_(p, 0, 0.1)
        assert m0(x).shape == (2, 168)
    p = sum(q.numel() for q in AFEncDec(44, n_raw=11, bridge=(1, 6, 24, 168),
                                        bridge_exact=3).parameters())
    print(f'encdec selftest ok; params = {p}')


if __name__ == '__main__':
    _selftest()