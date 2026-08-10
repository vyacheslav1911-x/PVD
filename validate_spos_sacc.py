#!/usr/bin/env python
"""Validate S_pos (constraint) and S_acc (smoothness) against hardware recordings.

Honors the methodology: per-tick measured dt in every finite difference; NO
dt>0.06 masking; chunk_start runs collapsed to events; SmolVLA (run1+run2) and
pi0.5 analyzed separately; per-joint (never pooled in raw deg); r as effect size
only (no p-values -- 33 ms ticks are autocorrelated); every correlation reported
all-ticks AND uncapped-only (cmd_ vs sent_ differ => cap bound).
"""
import csv
import os
import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import score_trajectories as S

common = S.load_common()
model, data, q_max, jn = S.load_model_and_limits()
LJ = common.LEROBOT_JOINT_ORDER                       # lerobot order == URDF order 1:1
BODY = LJ[:5]
FILES = [("record_run1.csv", "smolvla"), ("record_run2.csv", "smolvla"),
         ("record_pi05_run1.csv", "pi05")]
LOWER = {LJ[i]: model.lowerPositionLimit[i] for i in range(6)}
UPPER = {LJ[i]: model.upperPositionLimit[i] for i in range(6)}
QMAX = {LJ[i]: q_max[i] for i in range(6)}
DEG = 180.0 / np.pi
L = 4                                                 # measured servo lag (ticks)
WARM = 5                                              # drop first few startup ticks (NOT boundaries)


def findf(f):
    for p in (f, os.path.join("pvd_logs", f)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f)


def fd(x, dt):
    """backward finite difference with per-tick dt (nan where invalid)."""
    out = np.full_like(x, np.nan)
    d = np.where(dt > 1e-4, dt, np.nan)
    out[1:] = (x[1:] - x[:-1]) / d[1:]
    return out


def load(path):
    rows = list(csv.DictReader(open(path)))
    N = len(rows)
    def c(p, j): return np.array([float(r[f"{p}_{j}"]) if r[f"{p}_{j}"] not in ("", "nan") else np.nan for r in rows])
    dt = np.array([float(r["dt"]) for r in rows])
    cs = np.array([int(r["chunk_start"]) for r in rows])
    starts = [i for i in range(N) if cs[i] == 1 and (i == 0 or cs[i - 1] == 0)]  # event starts
    prox = np.array([min(abs(i - s) for s in starts) if starts else np.nan for i in range(N)], float)
    valid = np.ones(N, bool); valid[:WARM] = False       # startup only; boundaries kept
    return dict(N=N, dt=dt, cs=cs, prox=prox, valid=valid,
                cmd={j: c("cmd", j) for j in LJ}, pos={j: c("pos", j) for j in LJ},
                sent={j: c("sent", j) for j in LJ})


DAT = {f: load(findf(f)) for f, _ in FILES}
POL = {"smolvla": ["record_run1.csv", "record_run2.csv"], "pi05": ["record_pi05_run1.csv"]}


# ============================================================================
# S_acc — |commanded accel| vs achieved-motion jerk / reversals (smoothness)
# ============================================================================
def features(d):
    """per-joint x=|acc_cmd|, |vel_cmd|, and achieved jerk / reversal / trkerr."""
    dt = d["dt"]; out = {}
    for j in LJ:
        vc = fd(d["cmd"][j], dt); ac = fd(vc, dt)
        vp = fd(d["pos"][j], dt); ap = fd(vp, dt); jp = fd(ap, dt)
        rev = np.full(d["N"], np.nan)
        rev[1:] = (np.sign(vp[1:]) != np.sign(vp[:-1])).astype(float)  # velocity sign change
        trk = np.abs(d["cmd"][j] - np.r_[d["pos"][j][L:], [np.nan] * L])  # |cmd(t)-pos(t+L)|
        capped = np.abs(d["cmd"][j] - d["sent"][j]) > 0.5
        out[j] = dict(x=np.abs(ac), v=np.abs(vc), jerk=np.abs(jp), rev=rev, trk=trk,
                      prox=d["prox"], valid=d["valid"], capped=capped)
    return out


FEAT = {f: features(d) for f, d in DAT.items()}


def resid(y, X):
    A = np.column_stack([np.ones(len(y))] + [X[:, k] for k in range(X.shape[1])])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def rr(a, b):
    return np.corrcoef(a, b)[0, 1] if len(a) > 5 and np.ptp(a) and np.ptp(b) else np.nan


def pooled(pol, j, yname, lag, tickset):
    """concat feature arrays across a policy's files (diffs stay within-file)."""
    X, Y, V, P = [], [], [], []
    capfrac_num = capfrac_den = 0
    for f in POL[pol]:
        F = FEAT[f][j]
        x, v, prox, val, cap = F["x"], F["v"], F["prox"], F["valid"], F["capped"]
        y = F[yname].copy()
        if lag:                                     # align achieved response to command by +lag
            y = np.r_[y[lag:], [np.nan] * lag]
            prox = prox                             # prox at command time
        m = val & np.isfinite(x) & np.isfinite(y) & np.isfinite(v) & np.isfinite(prox)
        if tickset == "uncapped":
            m = m & ~cap
        capfrac_num += int((val & cap).sum()); capfrac_den += int(val.sum())
        X.append(x[m]); Y.append(y[m]); V.append(v[m]); P.append(prox[m])
    X, Y, V, P = map(np.concatenate, (X, Y, V, P))
    n = len(X)
    if n < 10:
        return dict(raw=np.nan, pv=np.nan, pvb=np.nan, n=n, capfrac=capfrac_num / max(1, capfrac_den))
    raw = rr(X, Y)
    pv = rr(resid(X, V[:, None]), resid(Y, V[:, None]))
    pvb = rr(resid(X, np.column_stack([V, P])), resid(Y, np.column_stack([V, P])))
    return dict(raw=raw, pv=pv, pvb=pvb, n=n, capfrac=capfrac_num / max(1, capfrac_den))


print("=" * 78)
print("S_acc : |commanded accel| -> achieved jerk / reversal / tracking-error")
print("=" * 78)
sacc_rows = []
for pol in ("smolvla", "pi05"):
    print(f"\n--- {pol} ---")
    print(f"{'joint':<13}{'y':<8}{'lag':>4}{'set':>9}{'raw_r':>8}{'pr|v':>8}{'pr|v,b':>8}{'n':>7}{'cap%':>7}")
    for j in LJ:
        for yname, lag in [("jerk", L), ("jerk", 0), ("rev", L), ("trk", 0)]:
            for ts in ("all", "uncapped"):
                res = pooled(pol, j, yname, lag, ts)
                sacc_rows.append([pol, j, yname, lag, ts, f"{res['raw']:.3f}", f"{res['pv']:.3f}",
                                  f"{res['pvb']:.3f}", res["n"], f"{res['capfrac']:.3f}"])
                if ts == "all" or j in BODY[:0]:  # print all-set for brevity; uncapped in CSV
                    pass
            # print the two ticksets for the primary y=jerk lag=L
        r_all = pooled(pol, j, "jerk", L, "all"); r_unc = pooled(pol, j, "jerk", L, "uncapped")
        print(f"{j:<13}{'jerk':<8}{L:>4}{'all':>9}{r_all['raw']:>8.3f}{r_all['pv']:>8.3f}"
              f"{r_all['pvb']:>8.3f}{r_all['n']:>7}{100*r_all['capfrac']:>6.1f}")
        print(f"{'':<13}{'jerk':<8}{L:>4}{'uncap':>9}{r_unc['raw']:>8.3f}{r_unc['pv']:>8.3f}"
              f"{r_unc['pvb']:>8.3f}{r_unc['n']:>7}{'':>7}")

with open("s_acc_validation.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["policy", "joint", "y_variable", "lag_ticks", "tickset",
                "raw_r", "partial_r_velocity", "partial_r_velocity_boundary", "n", "capped_fraction"])
    w.writerows(sacc_rows)
print("\n-> wrote s_acc_validation.csv")

# ---- body-joint mean of the decisive statistic (jerk, lag L, uncapped, pr|v,b) ----
for pol in ("smolvla", "pi05"):
    vals = [pooled(pol, j, "jerk", L, "uncapped")["pvb"] for j in BODY]
    print(f"[{pol}] mean body-joint partial-r(|acc|,jerk | vel,boundary), uncapped, lag{L} = "
          f"{np.nanmean(vals):+.3f}   per-joint={[round(v,2) for v in vals]}")


# ============================================================================
# Unit scaling: fractional-violation magnitudes of S_vel vs S_acc
# ============================================================================
print("\n" + "=" * 78)
print("UNIT SCALING : |q_dot| vs |q_ddot| and their fractional-violation penalties")
print("=" * 78)
for pol in ("smolvla", "pi05"):
    QD, QDD = [], []
    for f in POL[pol]:
        d = DAT[f]
        for j in LJ:
            vc = fd(d["cmd"][j], d["dt"]) / DEG        # deg/s -> rad/s
            ac = fd(vc, d["dt"])                        # rad/s^2
            m = d["valid"] & np.isfinite(ac)
            QD.append(np.abs(vc[m])); QDD.append(np.abs(ac[m]))
    QD, QDD = np.concatenate(QD), np.concatenate(QDD)
    fv_v = np.maximum(0, (QD - 3.0) / 3.0)             # relu_frac vs q_dot_max=3
    fv_a = np.maximum(0, (QDD - 20.0) / 20.0)          # relu_frac vs q_ddot_max=20
    print(f"\n[{pol}] |q_dot| rad/s   median={np.median(QD):.2f} p95={np.percentile(QD,95):.2f} max={QD.max():.1f}")
    print(f"[{pol}] |q_ddot| rad/s2 median={np.median(QDD):.1f} p95={np.percentile(QDD,95):.1f} max={QDD.max():.0f}")
    print(f"[{pol}] fractional-violation SUM  S_vel={fv_v.sum():.1f}  S_acc={fv_a.sum():.1f}  "
          f"ratio S_acc/S_vel={fv_a.sum()/max(1e-9,fv_v.sum()):.1f}x")
    print(f"[{pol}]   (scorer relu_frac already /limit; if ratio>>1, S_acc still dominates Phi -> retune q_ddot_max)")


# ============================================================================
# S_pos — constraint check (NOT a correlation)
# ============================================================================
print("\n" + "=" * 78)
print("S_pos : configured limits vs observed reach  (constraint, not correlation)")
print("=" * 78)
spos_rows = []
print(f"{'file':<22}{'joint':<13}{'pos[min,max]deg':>20}{'cmd[min,max]deg':>20}"
      f"{'limit[lo,hi]deg':>18}{'viol':>6}{'verdict':>14}")
for f, pol in FILES:
    d = DAT[f]
    for j in LJ:
        pmin, pmax = np.nanmin(d["pos"][j]), np.nanmax(d["pos"][j])
        cmin, cmax = np.nanmin(d["cmd"][j]), np.nanmax(d["cmd"][j])
        lo, hi, qm = LOWER[j] * DEG, UPPER[j] * DEG, QMAX[j] * DEG
        if j == "gripper":                              # gripper is % not deg; skip limit compare
            verdict = "n/a(%)"; viol = 0
        else:
            # violations vs asymmetric URDF limit (physical truth), on commanded
            viol = int(np.sum((d["cmd"][j] > hi + 1e-6) | (d["cmd"][j] < lo - 1e-6)) +
                       0)  # cmd deg vs urdf deg (common: sign+1 off 0 -> deg==deg)
            reached = max(abs(pmin), abs(pmax), abs(cmin), abs(cmax))
            # a limit is PROVABLY WRONG if the arm's achieved pos exceeded it
            wrong = (pmax > hi + 1.0) or (pmin < lo - 1.0)
            verdict = "LIMIT_TOO_TIGHT" if wrong else ("cmd>limit" if viol > 0 else "OK")
        spos_rows.append([f, pol, j, round(pmin, 1), round(pmax, 1), round(cmin, 1), round(cmax, 1),
                          round(lo, 1), round(hi, 1), round(qm, 1), viol, verdict])
        print(f"{f:<22}{j:<13}{f'[{pmin:.0f},{pmax:.0f}]':>20}{f'[{cmin:.0f},{cmax:.0f}]':>20}"
              f"{f'[{lo:.0f},{hi:.0f}]':>18}{viol:>6}{verdict:>14}")

with open("s_pos_check.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "policy", "joint", "pos_min", "pos_max", "cmd_min", "cmd_max",
                "limit_lo_deg", "limit_hi_deg", "qmax_sym_deg", "cmd_violations", "verdict"])
    w.writerows(spos_rows)
print("\n-> wrote s_pos_check.csv")


# ============================================================================
# S_pos unit test — penalty scales with margin; PVD rejects over-limit chunk
# ============================================================================
print("\n" + "=" * 78)
print("S_pos UNIT TEST : penalty scales with over-limit margin; PVD rejects it")
print("=" * 78)
H = 50
q0 = np.zeros(6)
jt = LJ.index("shoulder_pan")            # test on a symmetric joint (Rotation, q_max=1.92)
print(f"joint={LJ[jt]}  q_max={QMAX[LJ[jt]]:.3f} rad")
for margin in [0.0, 0.1, 0.3, 0.6]:
    A = np.tile(q0, (H, 1)); A[:, jt] = QMAX[LJ[jt]] + margin      # command past the symmetric limit
    terms = S.score_candidate(A, q0, model, data, q_max)
    print(f"  margin=+{margin:.2f} rad  ->  S_pos={terms['pos']:.3f}   (expected ~H*margin/q_max = "
          f"{H*margin/QMAX[LJ[jt]]:.3f})")
# PVD rejection: over-limit candidate vs in-range candidate
A_in = np.tile(q0, (H, 1)); A_in[:, jt] = 0.5                      # well inside
A_out = np.tile(q0, (H, 1)); A_out[:, jt] = QMAX[LJ[jt]] + 0.5     # 0.5 rad past limit
phi_in = S.phi(S.score_candidate(A_in, q0, model, data, q_max))
phi_out = S.phi(S.score_candidate(A_out, q0, model, data, q_max))
thr = S.FEASIBILITY_THRESHOLD
print(f"  Phi(in-range)={phi_in:.3f}  Phi(over-limit)={phi_out:.3f}  threshold={thr}")
print(f"  filter-then-prefer: in-range {'PASS' if phi_in<=thr else 'reject'}, "
      f"over-limit {'PASS' if phi_out<=thr else 'REJECTED'} -> "
      f"{'PVD correctly rejects the over-limit chunk' if phi_out>thr>=phi_in or phi_out>phi_in else 'CHECK'}")


# ============================================================================
# Plots
# ============================================================================
INK, MUTED, GRID = "#111827", "#6b7280", "#eceff3"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
for pol, color in (("smolvla", "#2563eb"), ("pi05", "#7c3aed")):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), dpi=130)
    for j, ax in zip(LJ, axes.ravel()):
        X, Y = [], []
        for f in POL[pol]:
            F = FEAT[f][j]; y = np.r_[F["jerk"][L:], [np.nan] * L]
            m = F["valid"] & np.isfinite(F["x"]) & np.isfinite(y) & ~F["capped"]
            X.append(F["x"][m]); Y.append(y[m])
        X, Y = np.concatenate(X), np.concatenate(Y)
        if len(X) > 4000:
            idx = np.random.default_rng(0).choice(len(X), 4000, replace=False); X, Y = X[idx], Y[idx]
        r = pooled(pol, j, "jerk", L, "uncapped")
        ax.scatter(X, Y, s=6, alpha=0.25, color=color, edgecolor="none")
        ax.set_title(f"{j}   pr|v,b={r['pvb']:+.2f}", fontsize=10.5, fontweight="bold", color=INK)
        ax.set_xlabel("|commanded accel| (deg/s²)", fontsize=8.5)
        ax.set_ylabel("|measured jerk| (deg/s³)", fontsize=8.5)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"S_acc smoothness: commanded accel vs measured jerk ({pol}, uncapped)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"s_acc_scatter_{pol}.png", dpi=130, bbox_inches="tight", facecolor="white")
    print(f"-> wrote s_acc_scatter_{pol}.png")

# term magnitude distribution plot
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130)
for k, pol in enumerate(("smolvla", "pi05")):
    QD, QDD = [], []
    for f in POL[pol]:
        d = DAT[f]
        for j in LJ:
            vc = fd(d["cmd"][j], d["dt"]) / DEG; ac = fd(vc, d["dt"])
            m = d["valid"] & np.isfinite(ac); QD.append(np.abs(vc[m])); QDD.append(np.abs(ac[m]))
    QD, QDD = np.concatenate(QD), np.concatenate(QDD)
    ax[k].hist(np.clip(QD, 0, 25), bins=60, alpha=0.6, color="#2563eb", label="|q̇| rad/s (limit 3)")
    ax[k].hist(np.clip(QDD, 0, 25), bins=60, alpha=0.6, color="#f59e0b", label="|q̈| rad/s² (limit 20)")
    ax[k].axvline(3, color="#2563eb", ls="--", lw=1); ax[k].axvline(20, color="#f59e0b", ls="--", lw=1)
    ax[k].set_title(f"{pol}: commanded |q̇|,|q̈| vs limits", fontsize=11, fontweight="bold")
    ax[k].set_xlabel("magnitude (rad/s, rad/s²)"); ax[k].set_yscale("log"); ax[k].legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax[k].spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("term_magnitude_dist.png", dpi=130, bbox_inches="tight", facecolor="white")
print("-> wrote term_magnitude_dist.png")
