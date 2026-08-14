"""phi_scorer.spectral -- standalone spectral term for fracE_bw.

DELIBERATELY DEPENDENCY-FREE (numpy only, no phi_scorer imports) so it can be unit
tested in isolation -- see test_spectral.py. The whole point of frac_E_bw is:

    "what fraction of a commanded joint trajectory's AC energy sits ABOVE the
     frequency the servo can physically track?"

Two conditioning choices matter and are made explicit here, because an earlier audit
showed the naive version double-counts artifacts:

  * DETREND. A 50-sample window that starts and ends at different angles has an edge
    discontinuity that leaks into every high bin. Mean-subtraction removes the DC bin
    but NOT the ramp, so it leaves that leakage in. LINEAR detrend removes the ramp
    and kills the leakage (a jitter-free min-jerk sweep drops from ~14% to ~0.1%).
    Default here is linear detrend.

  * WINDOW. A Hann window's main lobe is ~4 bins wide; with ~0.6 Hz bins a pure tone
    near the 1.3 Hz cutoff smears across it (a clean 1.3 Hz sinusoid reads ~29% with
    Hann vs ~3% without). So Hann is OPTIONAL and OFF by default; turn it on only if
    you specifically want lobe-shaping and accept the cross-cutoff bleed.

frac_E_bw = sum(P[f > f_cut]) / sum(P[f > 0])   per joint, then averaged across joints.
"""

import numpy as np


def joint_frac_above(x, dt, f_cut, detrend="linear", window="none"):
    """Fraction of a single joint signal's AC power above f_cut.

    x       : 1-D commanded trajectory for one joint (radians), length n.
    dt      : sample period (s). Bin spacing is 1/(n*dt).
    f_cut   : cutoff frequency (Hz) -- the servo bandwidth for this joint.
    detrend : "linear" (remove best-fit line -> no ramp leakage), "mean" (DC only),
              or "none".
    window  : "hann" or "none". None avoids main-lobe smearing across the cutoff.
    """
    n = len(x)
    x = np.asarray(x, dtype=float)
    # AC energy of the RAW signal -- used to detect the "signal was pure trend" case:
    # if detrending removes essentially all of it, the leftover is float noise and its
    # bin ratio is meaningless, so we must report 0 rather than garbage.
    ref_energy = float(np.sum((x - x.mean()) ** 2)) + 1e-30

    if detrend == "linear":
        t = np.arange(n)
        A = np.vstack([t, np.ones(n)]).T           # fit x ~ a*t + b, subtract it
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        x = x - A @ coef
    elif detrend == "mean":
        x = x - x.mean()
    # "none": leave as-is

    if window == "hann":
        x = x * np.hanning(n)

    freqs = np.fft.rfftfreq(n, dt)
    power = np.abs(np.fft.rfft(x)) ** 2

    ac = power[freqs > 0]                            # exclude the DC bin from the total
    total = ac.sum()
    # No real content: either the joint never moved, or detrending removed ~everything
    # (a pure trend). 1e-9 of the raw AC energy is a very loose floor -- it only trips
    # on float noise, never on a genuine (even tiny) oscillation.
    if total < 1e-9 * ref_energy:
        return 0.0
    above = power[freqs > f_cut].sum()
    return float(above / total)


def frac_E_bw(chunk, dt, bandwidth_hz, detrend="linear", window="none"):
    """Average over joints of joint_frac_above.

    chunk        : [T, J] commanded trajectory in radians.
    dt           : sample period (s).
    bandwidth_hz : length-J array/list of per-joint cutoff frequencies (Hz).
    Returns a single scalar in [0, 1], or nan if no joint carries any energy.
    """
    chunk = np.asarray(chunk, dtype=float)
    T, J = chunk.shape
    fr = []
    for j in range(J):
        f = joint_frac_above(chunk[:, j], dt, float(bandwidth_hz[j]), detrend, window)
        # joints that are perfectly still contribute no meaningful fraction; skip them
        if np.ptp(chunk[:, j]) > 0:
            fr.append(f)
    return float(np.mean(fr)) if fr else np.nan
