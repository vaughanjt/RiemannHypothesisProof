"""Fit r(T) trajectory from all computed data points."""
import numpy as np
from scipy.optimize import curve_fit

# All data points: (T, r)
data = [
    (394, 0.9263),
    (1062, 0.9054),
    (2620, 0.8809),
    (5541, 0.8660),
    (9963, 0.8528),
    (24315, 0.8064),
    (47602, 0.8087),
    (94811, 0.8207),
    (2.676e11, 0.6302),
]

T = np.array([d[0] for d in data])
r = np.array([d[1] for d in data])
logT = np.log(T)

print("="*70)
print("r(T) TRAJECTORY — FULL DATA")
print("="*70)
print(f"\n  {'T':>14} {'log(T)':>8} {'r':>8}")
print(f"  {'-'*34}")
for Ti, ri in data:
    print(f"  {Ti:>14.0f} {np.log(Ti):>8.2f} {ri:>+8.4f}")

# Model 1: r = a - b * log(T)
print(f"\n{'='*70}")
print("MODEL FITS")
print("="*70)

def model_linear(T, a, b):
    return a - b * np.log(T)

def model_asymptotic(T, r_inf, a):
    return r_inf + a / np.log(T)

def model_loglog(T, a, b):
    return a - b * np.log(np.log(T))

def model_power(T, r_inf, a, alpha):
    return r_inf + a / np.log(T)**alpha

models = [
    ("r = a - b*log(T)", model_linear, [1.0, 0.01], "r -> -inf (eventually)"),
    ("r = r_inf + a/log(T)", model_asymptotic, [0.5, 2.0], "r -> r_inf"),
    ("r = a - b*log(log(T))", model_loglog, [1.3, 0.2], "r -> -inf (very slowly)"),
]

for name, func, p0, behavior in models:
    try:
        popt, pcov = curve_fit(func, T, r, p0=p0, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        residuals = r - func(T, *popt)
        rmse = np.sqrt(np.mean(residuals**2))

        print(f"\n  {name}")
        print(f"    Params: {', '.join(f'{p:.6f} +/- {e:.6f}' for p, e in zip(popt, perr))}")
        print(f"    RMSE: {rmse:.6f}")
        print(f"    Behavior: {behavior}")

        print(f"    Extrapolations:")
        for T_ext in [1e6, 1e8, 1e11, 1e15, 1e20, 1e30]:
            r_ext = func(T_ext, *popt)
            f_bound = max(0, (1-r_ext)/(1+r_ext)) if r_ext < 1 else 0
            on_line = (1-f_bound)*100
            print(f"      T = {T_ext:.0e}: r = {r_ext:+.4f}, "
                  f"f <= {f_bound:.3f} ({on_line:.1f}% on-line)")

    except Exception as e:
        print(f"\n  {name}: FAILED ({e})")

# Try the power model with 3 params
try:
    popt, pcov = curve_fit(model_power, T, r, p0=[0.4, 2.0, 1.0], maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    residuals = r - model_power(T, *popt)
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"\n  r = r_inf + a/log(T)^alpha")
    print(f"    r_inf = {popt[0]:.4f} +/- {perr[0]:.4f}")
    print(f"    a = {popt[1]:.4f} +/- {perr[1]:.4f}")
    print(f"    alpha = {popt[2]:.4f} +/- {perr[2]:.4f}")
    print(f"    RMSE: {rmse:.6f}")

    print(f"    Extrapolations:")
    for T_ext in [1e6, 1e8, 1e11, 1e15, 1e20, 1e30]:
        r_ext = model_power(T_ext, *popt)
        f_bound = max(0, (1-r_ext)/(1+r_ext)) if r_ext < 1 else 0
        on_line = (1-f_bound)*100
        print(f"      T = {T_ext:.0e}: r = {r_ext:+.4f}, "
              f"f <= {f_bound:.3f} ({on_line:.1f}% on-line)")
except Exception as e:
    print(f"\n  Power model: FAILED ({e})")


# ============================================================
# COMPARISON WITH CONREY
# ============================================================
print(f"\n{'='*70}")
print("COMPARISON WITH CONREY'S 40.1%")
print("="*70)

print(f"""
  Conrey (1989): at least 40.1% on the critical line (unconditional theorem)

  Our approach (density bound f <= (1-r)/(1+r)):
    - Valid as a numerical observation, not yet a theorem
    - At T ~ 400: r = 0.93, f <= 3.7%, at least 96.3% on-line
    - At T ~ 95000: r = 0.82, f <= 9.9%, at least 90.1% on-line
    - At T ~ 2.68e11: r = 0.63, f <= 22.7%, at least 77.3% on-line

  Even at T ~ 10^11, our density bound DOUBLES Conrey's result.

  The question: does r have a positive limit r_inf > 0?
  If r_inf > 0, the bound f <= (1-r_inf)/(1+r_inf) holds for ALL T.
  If r_inf = 0, the bound degenerates (but very slowly).
""")

# What r_inf gives Conrey's bound?
# (1-r)/(1+r) = 0.599 => 1-r = 0.599 + 0.599r => 1 = 0.599 + 1.599r => r = 0.401/1.599 = 0.2508
print(f"  To match Conrey's 40.1%, we need r >= {0.401/1.599:.4f}")
print(f"  Our lowest measured r = 0.630 (at T ~ 2.68e11)")
print(f"  Margin above Conrey threshold: {0.630 - 0.401/1.599:.4f}")
