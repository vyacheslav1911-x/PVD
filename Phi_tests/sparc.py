"""SPARC — spectral arc length smoothness metric.

lambda_S is the negative arc length of the normalised magnitude spectrum of a
speed profile, measured over an adaptive band [0, f_cut] where f_cut is the
last frequency (below FC) at which the spectrum is still above AMP_TH of its
own maximum.

Smoother movement -> spectrum concentrated near DC -> shorter arc -> lambda_S
closer to 0.  Dithery movement -> broad, rippled spectrum -> long arc ->
lambda_S more negative.

Normalising the magnitude by its own max makes the score invariant to the
amplitude of the movement; normalising the frequency step by the width of the
selected band makes it invariant to its duration.
"""

import numpy as np

FC = 10.0        # frequency cap, Hz
AMP_TH = 0.05    # adaptive-band amplitude threshold, fraction of max
PAD_EXP = 4      # zero-pad to 2 ** (ceil(log2(N)) + PAD_EXP)


def sparc(speed, fs, fc=FC, amp_th=AMP_TH, pad_exp=PAD_EXP):
    """Return lambda_S for a speed profile sampled uniformly at fs Hz."""
    speed = np.asarray(speed, dtype=float)
    nfft = 2 ** (int(np.ceil(np.log2(speed.size))) + pad_exp)

    Mf = np.abs(np.fft.fft(speed, nfft))
    if Mf.max() == 0:
        return np.nan
    Mf = Mf / Mf.max()

    f = np.arange(0, fs, fs / nfft)[:nfft]
    keep = f <= fc
    f, Mf = f[keep], Mf[keep]

    above = np.where(Mf >= amp_th)[0]
    if above.size < 2:
        return np.nan
    f_sel = f[above[0] : above[-1] + 1]
    M_sel = Mf[above[0] : above[-1] + 1]

    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    return -np.sum(np.sqrt(df**2 + np.diff(M_sel) ** 2))


def sparc_cutoff(speed, fs, fc=FC, amp_th=AMP_TH, pad_exp=PAD_EXP):
    """The adaptive band edge f_cut that sparc() ended up using, in Hz."""
    speed = np.asarray(speed, dtype=float)
    nfft = 2 ** (int(np.ceil(np.log2(speed.size))) + pad_exp)
    Mf = np.abs(np.fft.fft(speed, nfft))
    Mf = Mf / Mf.max()
    f = np.arange(0, fs, fs / nfft)[:nfft]
    keep = f <= fc
    f, Mf = f[keep], Mf[keep]
    return f[np.where(Mf >= amp_th)[0][-1]]


# --------------------------------------------------------------- synthetics
def min_jerk_speed(duration, fs, amplitude=1.0):
    """Speed profile of a minimum-jerk movement: 30 tau^2 (1 - tau)^2."""
    n = int(round(duration * fs))
    tau = np.linspace(0.0, 1.0, n, endpoint=False)
    return amplitude * 30.0 * tau**2 * (1.0 - tau) ** 2


def overlapping_bumps(fs, n_bumps=5, width=0.4, hop=0.4, amplitude=1.0):
    """n_bumps min-jerk bells, each `width` long, started every `hop` seconds."""
    total = int(round((hop * (n_bumps - 1) + width) * fs))
    out = np.zeros(total)
    bump = min_jerk_speed(width, fs, amplitude)
    for k in range(n_bumps):
        i = int(round(k * hop * fs))
        out[i : i + bump.size] += bump[: total - i]
    return out
