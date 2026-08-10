"""Section 4 -- per-joint forward servo model, identified from the ramp logs.

The survey proposes FOPDT/SOPDT + dead time. But the ramp measurements showed a
LINEAR velocity ramp to a flat plateau, which is the signature of a rate-limited
(trapezoidal-profile) actuator, not of a linear second-order system. The STS3215
runs an internal profile governed by its Maximum_Acceleration register, so the
large-signal response is nonlinear by construction.

Both models are therefore fitted and compared on the same data:

  LIN   second-order + dead time:  ydd = wn^2 (u(t-Td) - y) - 2 z wn yd
  RATE  rate-limited tracker:      yd  saturated to +-vmax,
                                   ydd saturated to +-amax, toward u(t-Td)

Identification uses the 1.7 kHz ramp step responses (clean, saturating steps).
Validation is on HELD-OUT data the fit never saw: the 30 Hz plain replay
(exec_plain.csv), which is a real policy trajectory with goal_ and pos_ logged.
"""

import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REPO = os.path.expanduser("~/Desktop/PVD")
RAMP = os.path.join(REPO, "pvd_logs", "ramp_{}.csv")
CALIB = os.path.expanduser(
    "~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json")
T2R = 2 * np.pi / 4096
D2R = np.pi / 180
GRIP = (1.74533 + 0.174533) / 100


def sim_linear(u, dt, wn, z, td, y0=None):
    n = len(u)
    d = int(round(td / dt))
    cur = float(u[0] if y0 is None else y0)   # MUST start at the real initial
    y = np.empty(n); yd = 0.0; y[0] = cur     # position, not at the goal
    for k in range(1, n):
        tgt = u[max(0, k - d)]
        ydd = wn * wn * (tgt - cur) - 2 * z * wn * yd
        yd += ydd * dt
        cur += yd * dt
        y[k] = cur
    return y


def sim_rate(u, dt, vmax, amax, kp, td, y0=None):
    n = len(u)
    d = int(round(td / dt))
    cur = float(u[0] if y0 is None else y0)
    y = np.empty(n); yd = 0.0; y[0] = cur
    for k in range(1, n):
        tgt = u[max(0, k - d)]
        v_des = np.clip(kp * (tgt - cur), -vmax, vmax)
        dv = np.clip(v_des - yd, -amax * dt, amax * dt)
        yd += dv
        cur += yd * dt
        y[k] = cur
    return y


def fit_joint(j):
    rows = list(csv.reader(open(RAMP.format(j))))[1:]
    a = np.array([[float(x) for x in r[1:]] for r in rows])
    tags = np.array([r[0] for r in rows])
    m = tags == "pos"
    t, pos, goal = a[m, 0], a[m, 1] * T2R, a[m, 3] * T2R
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1], dt)
    y = np.interp(tu, t, pos)
    u = np.interp(tu, t, goal)

    def err_lin(p):
        wn, z, td = np.exp(p[0]), 1 / (1 + np.exp(-p[1])) * 2, abs(p[2])
        return np.sqrt(np.mean((sim_linear(u, dt, wn, z, td, y[0]) - y) ** 2))

    def err_rate(p):
        vmax, amax, kp, td = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), abs(p[3])
        return np.sqrt(np.mean((sim_rate(u, dt, vmax, amax, kp, td, y[0]) - y) ** 2))

    rl = minimize(err_lin, [np.log(15.0), 0.0, 0.01], method="Nelder-Mead",
                  options=dict(maxiter=600, xatol=1e-4, fatol=1e-8))
    rr = minimize(err_rate, [np.log(4.0), np.log(28.0), np.log(20.0), 0.01],
                  method="Nelder-Mead", options=dict(maxiter=1200, xatol=1e-4, fatol=1e-8))
    lin = dict(wn=float(np.exp(rl.x[0])), z=float(1 / (1 + np.exp(-rl.x[1])) * 2),
               td=float(abs(rl.x[2])), rmse=float(rl.fun))
    rate = dict(vmax=float(np.exp(rr.x[0])), amax=float(np.exp(rr.x[1])),
                kp=float(np.exp(rr.x[2])), td=float(abs(rr.x[3])), rmse=float(rr.fun))
    span = float(np.ptp(y))
    return lin, rate, span, dt


def main():
    print("IDENTIFICATION on the 1.7 kHz ramp step responses")
    print(f"{'joint':<15}{'LIN wn':>8}{'zeta':>7}{'Td ms':>7}{'RMSE deg':>10}   "
          f"{'RATE vmax':>10}{'amax':>8}{'kp':>7}{'Td ms':>7}{'RMSE deg':>10}{'  winner':>9}")
    models = {}
    for j in JOINTS:
        lin, rate, span, dt = fit_joint(j)
        models[j] = dict(lin=lin, rate=rate, dt=dt)
        win = "RATE" if rate["rmse"] < lin["rmse"] else "LIN"
        print(f"{j:<15}{lin['wn']:>8.1f}{lin['z']:>7.2f}{1000*lin['td']:>7.1f}"
              f"{lin['rmse']/D2R:>10.3f}   {rate['vmax']:>10.2f}{rate['amax']:>8.1f}"
              f"{rate['kp']:>7.1f}{1000*rate['td']:>7.1f}{rate['rmse']/D2R:>10.3f}{win:>9}")

    json.dump(models, open(os.path.join(REPO, "pvd_logs/servo_models.json"), "w"), indent=1)

    # ---------------- validation on HELD-OUT policy trajectory ---------------
    cal = json.load(open(CALIB))

    def raw2rad(j, raw):
        lo, hi = cal[j]["range_min"], cal[j]["range_max"]; mid = (lo + hi) / 2
        if j == "gripper":
            return ((raw - lo) / (hi - lo) * 100.0) * GRIP
        return (raw - mid) * 360 / 4095 * D2R

    e = pd.read_csv(os.path.join(REPO, "pvd_logs/exec_plain.csv"))
    te = e["t"].to_numpy()
    dt_v = float(np.median(np.diff(te)))
    tu = np.arange(te[0], te[-1], dt_v)
    print(f"\nVALIDATION on held-out exec_plain.csv ({len(tu)} samples @ "
          f"{1/dt_v:.0f} Hz) -- the fit never saw this data")
    print(f"{'joint':<15}{'actual RMSE':>13}{'LIN pred':>11}{'RATE pred':>11}"
          f"{'LIN err':>10}{'RATE err':>10}")
    tot = {"lin": [], "rate": [], "act": []}
    for j in JOINTS:
        u = np.interp(tu, te, raw2rad(j, e[f"goal_{j}"].to_numpy(float)))
        y = np.interp(tu, te, raw2rad(j, e[f"pos_{j}"].to_numpy(float)))
        M = models[j]
        yl = sim_linear(u, dt_v, M["lin"]["wn"], M["lin"]["z"], M["lin"]["td"], y[0])
        yr = sim_rate(u, dt_v, M["rate"]["vmax"], M["rate"]["amax"],
                      M["rate"]["kp"], M["rate"]["td"], y[0])
        act = np.sqrt(np.mean((u - y) ** 2))
        pl = np.sqrt(np.mean((u - yl) ** 2))
        pr = np.sqrt(np.mean((u - yr) ** 2))
        el = np.sqrt(np.mean((yl - y) ** 2))
        er = np.sqrt(np.mean((yr - y) ** 2))
        tot["act"].append(act); tot["lin"].append(pl); tot["rate"].append(pr)
        print(f"{j:<15}{act/D2R:>13.3f}{pl/D2R:>11.3f}{pr/D2R:>11.3f}"
              f"{el/D2R:>10.3f}{er/D2R:>10.3f}")
    print(f"{'MEAN':<15}{np.mean(tot['act'])/D2R:>13.3f}"
          f"{np.mean(tot['lin'])/D2R:>11.3f}{np.mean(tot['rate'])/D2R:>11.3f}")
    print("\n'actual RMSE' = real tracking error (goal vs achieved).")
    print("'LIN/RATE pred' = tracking error the model PREDICTS.")
    print("'LIN/RATE err' = how far the model's trajectory is from the real one.")


if __name__ == "__main__":
    main()
