"""Step 3 — sanity-test sparc() on synthetic data before touching a rollout.

Expected:  single min-jerk bell ~ -1.4,  five overlapping bumps ~ -3.4.
Invariance: scaling the bell by 10x and stretching it to 3 s must not move
the score.
"""

import numpy as np

from sparc import min_jerk_speed, overlapping_bumps, sparc, sparc_cutoff

FS = 30.0
TOL = 0.05  # how much an invariance test is allowed to drift

bell = min_jerk_speed(1.0, FS)
bumps = overlapping_bumps(FS, width=0.4, hop=0.4)

l_bell = sparc(bell, FS)
l_bumps = sparc(bumps, FS)

print("reference signals")
print(f"  min-jerk bell, 1 s      n={bell.size:3d}  f_cut {sparc_cutoff(bell, FS):5.2f} Hz"
      f"   lambda_S = {l_bell:+.3f}   (expect ~ -1.4)")
print(f"  five bumps, 0% overlap  n={bumps.size:3d}  f_cut {sparc_cutoff(bumps, FS):5.2f} Hz"
      f"   lambda_S = {l_bumps:+.3f}   (expect ~ -3.4)")
print(f"  separation              {l_bell - l_bumps:+.3f}")

# -3.4 only shows up once the bumps are distinct; overlap merges them back
# into something the metric reads as nearly as smooth as a single bell.
print("\n  sensitivity to bump overlap (width 0.4 s)")
for hop in (0.15, 0.20, 0.25, 0.30, 0.40):
    s = overlapping_bumps(FS, width=0.4, hop=hop)
    print(f"    hop {hop:.2f} s  overlap {100 * max(0, 0.4 - hop) / 0.4:3.0f}%"
          f"   lambda_S = {sparc(s, FS):+.3f}")

print("\ninvariance")
checks = []
for name, sig in [
    ("amplitude x10", min_jerk_speed(1.0, FS, amplitude=10.0)),
    ("amplitude x0.01", min_jerk_speed(1.0, FS, amplitude=0.01)),
    ("duration 3 s", min_jerk_speed(3.0, FS)),
    ("duration 0.5 s", min_jerk_speed(0.5, FS)),
    ("bumps x10", overlapping_bumps(FS, width=0.4, hop=0.4, amplitude=10.0)),
    ("bumps stretched", overlapping_bumps(FS, width=1.2, hop=1.2)),
]:
    l = sparc(sig, FS)
    d = l - (l_bumps if name.startswith("bumps") else l_bell)
    ok = abs(d) < TOL
    checks.append(ok)
    print(f"  {name:16s} lambda_S = {l:+.3f}   delta {d:+.4f}   "
          f"{'ok' if ok else 'MOVED'}")

# the metric must also order these correctly, which is the point of it
order_ok = l_bell > l_bumps
checks.append(order_ok)
print(f"\n  bell scores above bumps: {'ok' if order_ok else 'FAILED'}")

print("\n" + ("all checks passed" if all(checks) else "SOME CHECKS FAILED"))
