"""Unit tests pinning the fracE_bw conditioning (the artifacts from the earlier audit).

Run:  python -m phi_scorer.test_spectral   (or pytest phi_scorer/test_spectral.py)

They lock in WHY the defaults are linear-detrend + no-window:
  1. A jitter-free min-jerk sweep must read ~0 above cutoff once linearly detrended
     (mean-subtraction alone leaves ramp leakage -> a large false reading).
  2. A pure cutoff-frequency tone must NOT be smeared across the cutoff by a Hann
     window (Hann inflates it; no-window keeps it clean).
"""
import numpy as np

from .spectral import joint_frac_above

FS = 29.94
DT = 1.0 / FS
N = 50
F_CUT = 1.3


def test_trend_leakage_needs_linear_detrend():
    # A pure linear ramp is the cleanest case: it is ONLY trend, no real oscillation.
    # Mean-subtraction leaves the ramp -> its periodic discontinuity leaks into every
    # high bin (false high-frequency energy). Linear detrend removes the line exactly,
    # leaving ~zero -> the "signal" correctly reads as having no genuine content.
    x = np.linspace(0.0, 1.0, N)
    frac_mean = joint_frac_above(x, DT, F_CUT, detrend="mean", window="none")
    frac_lin = joint_frac_above(x, DT, F_CUT, detrend="linear", window="none")
    assert frac_mean > 0.02, f"a ramp should leak under mean-detrend, got {frac_mean:.4f}"
    assert frac_lin < 0.01, f"linear detrend should remove the ramp, got {frac_lin:.4f}"
    assert frac_lin < frac_mean, "linear detrend must reduce the leakage vs mean-detrend"
    print(f"[ok] trend leakage: mean={frac_mean:.3f} -> linear={frac_lin:.3f}")


def test_hann_smears_a_pure_tone_across_cutoff():
    # A clean tone at a BIN CENTRE just BELOW the cutoff (bin 2 = 2*fs/N = 1.198 Hz < 1.3).
    # With no window its energy is entirely in that sub-cutoff bin -> ~0 above cutoff.
    # A Hann window's ~4-bin main lobe spreads it into bins above the cutoff -> inflated.
    f_bin2 = 2 * FS / N                               # 1.198 Hz, below F_CUT
    t = np.arange(N) * DT
    x = np.sin(2 * np.pi * f_bin2 * t)
    frac_none = joint_frac_above(x, DT, F_CUT, detrend="linear", window="none")
    frac_hann = joint_frac_above(x, DT, F_CUT, detrend="linear", window="hann")
    assert frac_none < 0.06, f"a sub-cutoff bin tone should read ~0 no-window, got {frac_none:.3f}"
    assert frac_hann > 2 * frac_none and frac_hann > frac_none + 0.05, \
        f"Hann should smear it across the cutoff: none={frac_none:.3f} hann={frac_hann:.3f}"
    print(f"[ok] hann smearing: none={frac_none:.3f} << hann={frac_hann:.3f}")


def test_still_joint_is_zero():
    assert joint_frac_above(np.ones(N) * 0.7, DT, F_CUT) == 0.0
    print("[ok] a motionless joint has zero fraction")


if __name__ == "__main__":
    test_trend_leakage_needs_linear_detrend()
    test_hann_smears_a_pure_tone_across_cutoff()
    test_still_joint_is_zero()
    print("\nall spectral tests passed")
