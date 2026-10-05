"""The two Autoformer mechanisms (Wu et al., 2021), in a small form.

SeriesDecomp  x -> (seasonal, trend), trend = centred moving average, ends replicated.
AutoCorrelation
              R(tau) = sum_t q_t k_{t-tau} for every delay tau at once (FFT).  The best
              KTOP delays inside the band [0, KMAX) are kept, softmax(R/theta) turns their scores
              into weights, and the output is sum_j w_j v_{t - tau_j} with circular indexing.

Two fixed design choices, both ablated in the report:
  KTOP = 25  how many delays are aggregated.  The reference uses ceil(3 ln T); for T = 168 that is 15.
  KMAX = T/2 a delay longer than half the 168-step ring is mostly wrap-around, so delays past half
              the ring are masked out (-inf) before the top-k is taken.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

KTOP = 25                      # number of aggregated delays
KMIN, KMAX = 0, 84             # delay band for T = 168 (KMAX = T/2)


class MovingAvg(nn.Module):
    """Centred moving average; the ends are padded by repeating the first/last value."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):                                       # [B, T, C]
        p = (self.k - 1) // 2
        xp = torch.cat([x[:, :1].expand(-1, p, -1), x, x[:, -1:].expand(-1, self.k - 1 - p, -1)], 1)
        return F.avg_pool1d(xp.transpose(1, 2), self.k, 1).transpose(1, 2)


class SeriesDecomp(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.ma = MovingAvg(k)

    def forward(self, x):
        t = self.ma(x)
        return x - t, t


class AutoCorrelation(nn.Module):
    def __init__(self, d, heads, drop=0.1, ktop=KTOP, kmin=KMIN, kmax=KMAX):
        super().__init__()
        self.h, self.ktop, self.kmin, self.kmax = heads, int(ktop), int(kmin), int(kmax)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(drop)
        self.log_temp = nn.Parameter(torch.zeros(()))         # learned temperature, theta = softplus(.)

    def forward(self, x):                                       # [B, T, d]
        B, T, d = x.shape
        dh = d // self.h
        split = lambda z: z.view(B, T, self.h, dh).permute(0, 2, 3, 1)          # [B, heads, dh, T]
        Q, K, V = split(self.q(x)), split(self.k(x)), split(self.v(x))
        R = torch.fft.irfft(torch.fft.rfft(Q, dim=-1) * torch.conj(torch.fft.rfft(K, dim=-1)),
                            n=T, dim=-1)                       # [B, heads, dh, T]
        R = R.mean((1, 2))                                      # one score per delay, shared by all heads
        if self.kmin or self.kmax:
            R = R + torch.where(torch.arange(T, device=x.device) < self.kmax, 0.0, -float('inf'))
        val, tau = R.topk(min(self.ktop, T), dim=-1)
        w = torch.softmax(val / (F.softplus(self.log_temp) + 1e-3), dim=-1)
        # sum_j w_j v_{t-tau_j} is a circular convolution of v with a kernel that is w_j at tau_j
        omega = torch.zeros(B, T, device=x.device, dtype=w.dtype).scatter(1, tau, w)
        out = torch.fft.irfft(torch.fft.rfft(V.reshape(B, d, T), dim=-1)
                              * torch.fft.rfft(omega, dim=-1)[:, None, :], n=T, dim=-1)
        return self.o(self.drop(out.transpose(1, 2)))


class FFN(nn.Module):
    def __init__(self, d, mult=2, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(mult * d, d))

    def forward(self, x):
        return self.net(x)


def brute_force_R(q, k):
    """Equation (4) computed the slow way, used only by the self-test. q, k: [B, T, dh]."""
    T = q.shape[1]
    return torch.stack([(q * torch.roll(k, tau, dims=1)).sum(1).mean(-1) for tau in range(T)], 1)


def _selftest():
    torch.manual_seed(0)
    B, T, d, h = 3, 37, 8, 2
    q = torch.randn(B, T, d // h)
    a = brute_force_R(q, q)
    qh = q.transpose(1, 2).unsqueeze(1)                      # [B, 1, dh, T], as inside the layer
    R = torch.fft.irfft(torch.fft.rfft(qh, dim=-1) * torch.conj(torch.fft.rfft(qh, dim=-1)),
                        n=T, dim=-1).mean((1, 2))
    assert R.shape == a.shape and (a - R).abs().max() < 1e-4, (a - R).abs().max()
    # decomposition adds back up and keeps the length
    s, t = SeriesDecomp(9)(torch.randn(2, T, 4))
    x = torch.randn(2, T, 4)
    assert (SeriesDecomp(9)(x)[0] + SeriesDecomp(9)(x)[1] - x).abs().max() < 1e-5
    assert SeriesDecomp(9)(x)[0].shape == x.shape
    # the layer runs and keeps the shape
    assert AutoCorrelation(d, h)(torch.randn(2, T, d)).shape == (2, T, d)
    print('blocks selftest ok')


if __name__ == '__main__':
    _selftest()
