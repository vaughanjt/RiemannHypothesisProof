"""
Session 25j: Verify the Bernstein ellipse prediction.

Grok's corrected proof claims:
  - Nearest digamma pole at Im(z) = L/4
  - Bernstein ellipse parameter rho(L) = exp(pi*L/4)
  - Singular value decay: sigma_k <= C * rho^{-k} = C * exp(-pi*k*L/4)
  - For k = N+1 = 31: |eps_0| <= C * exp(-31*pi*L/4)

Check: does this predict the measured c' ~ 43?
  c' = 31 * pi / 4 = 24.3 * L ... wait, that's per unit L.
  Actually: |eps_0| <= C * exp(-c*N*L) with c = pi/4 ~ 0.785
  So c' = c * N = 0.785 * 31 = 24.3

But we measured c' ~ 43 (from log10|eps_0| ~ -18.8*L).
  c'_measured = 18.8 * ln(10) = 43.3

Prediction: c' = pi*N/4 = pi*31/4 = 24.3
Measured:    c' = 43.3

The prediction is OFF BY ~2x. Let's check if the pole location is correct
and whether there are better bounds.

Also verify: the digamma pole is at z = iL(m+1/4)/pi, and for
the archimedean PLUS prime contributions in b_n, the actual analyticity
strip might be wider than L/4.
"""

import mpmath
from mpmath import mp, mpf, pi, log, nstr
import numpy as np

mp.dps = 30

print("BERNSTEIN ELLIPSE PREDICTION CHECK")
print("=" * 70)

# Predicted rate from Grok's skeleton
c_per_mode = float(pi) / 4  # pi/4 ~ 0.785
print(f"c per mode (pi/4): {c_per_mode:.4f}")
print()

# For various N and L
for lam_sq, eps_measured_log10 in [(14, -49.2), (30, -65.8), (50, -73.4)]:
    L = float(log(mpf(lam_sq)))
    N = 30
    k = N + 1  # = 31 (smallest singular value index)

    # Predicted: |eps_0| <= C * exp(-c * k * L) = C * exp(-pi/4 * 31 * L)
    c_prime_predicted = c_per_mode * k  # effective c' = pi*31/4 ~ 24.3
    log10_predicted = -c_prime_predicted * L / np.log(10)

    # Measured
    c_prime_measured = -eps_measured_log10 * np.log(10) / L

    print(f"lam^2={lam_sq:>4}, L={L:.3f}:")
    print(f"  Predicted log10|eps_0| = {log10_predicted:.1f}  (c'={c_prime_predicted:.1f})")
    print(f"  Measured  log10|eps_0| = {eps_measured_log10:.1f}  (c'={c_prime_measured:.1f})")
    print(f"  Ratio measured/predicted: {eps_measured_log10/log10_predicted:.2f}")
    print()

# The digamma pole analysis
print("DIGAMMA POLE ANALYSIS")
print("-" * 70)
print("psi(pi*i*z/L + 1/4) has poles at pi*i*z/L + 1/4 = -m (m=0,1,...)")
print("=> z = iL(m + 1/4)/pi")
print()
for lam_sq in [14, 50, 100]:
    L = float(log(mpf(lam_sq)))
    # Nearest pole: m=0, z = iL/(4pi)... wait
    # z = iL(0 + 1/4)/pi = iL/(4pi)
    # Hmm, but Grok wrote z = iL(m + 1/4) without the pi...
    # Let me recheck: pi*i*z/L + 1/4 = 0 => z = L/(4*pi*i) = -iL/(4pi)
    # So Im(z) = -L/(4pi), and |Im(z)| = L/(4pi)

    pole_grok = L / 4  # Grok claims Im(z) = L/4
    pole_actual = L / (4 * np.pi)  # From the equation: Im(z) = L/(4pi)

    print(f"lam^2={lam_sq:>4}, L={L:.3f}:")
    print(f"  Grok's pole:  Im(z) = L/4 = {pole_grok:.4f}")
    print(f"  Actual pole:  Im(z) = L/(4pi) = {pole_actual:.4f}")
    print(f"  Ratio: {pole_grok/pole_actual:.4f} (Grok is off by factor pi)")
    print()

print("CORRECTION:")
print("The digamma pole equation is:")
print("  pi*i*z/L + 1/4 = -m  =>  z = -iL(m+1/4)/pi")
print("  |Im(z)| = L(m+1/4)/pi, nearest (m=0) = L/(4pi)")
print()
print("Grok wrote z = iL(m+1/4) — MISSING the 1/pi factor!")
print("Correct rho = exp(pi * L/(4pi)) = exp(L/4)  (not exp(pi*L/4))")
print()

# With corrected rho:
print("WITH CORRECTED rho = exp(L/4):")
c_corrected = 1.0 / 4  # L/4 per mode
for lam_sq, eps_log10 in [(14, -49.2), (30, -65.8), (50, -73.4)]:
    L = float(log(mpf(lam_sq)))
    k = 31
    c_prime = c_corrected * k  # = 31/4 = 7.75
    log10_pred = -c_prime * L / np.log(10)
    c_prime_meas = -eps_log10 * np.log(10) / L
    print(f"  lam^2={lam_sq}: predicted log10 = {log10_pred:.1f}, measured = {eps_log10:.1f}, "
          f"ratio = {eps_log10/log10_pred:.2f}")

print()
print("Still off. The actual analyticity radius may be wider due to")
print("the W_p (prime) terms also extending the strip, or the")
print("Beckermann bound having a different constant.")
print()
print("EMPIRICAL FIT: what rho matches the data?")
for lam_sq, eps_log10 in [(14, -49.2), (30, -65.8), (50, -73.4)]:
    L = float(log(mpf(lam_sq)))
    k = 31  # if eps_0 = sigma_{N+1}
    # sigma_k <= C * rho^{-k}
    # log10|eps_0| = log10(C) - k * log10(rho)
    # Assuming log10(C) ~ 0: rho ~ 10^{-eps_log10/k}
    rho_empirical = 10 ** (-eps_log10 / k)
    log_rho = np.log(rho_empirical)
    # Bernstein: rho = exp(d) where d = analyticity half-width / N
    # For our case: d = log(rho) = -eps_log10*ln(10)/k
    d_empirical = log_rho
    analyticity_width = d_empirical * 30  # times N since Bernstein ellipse scales with N
    print(f"  lam^2={lam_sq}: rho = {rho_empirical:.2f}, log(rho) = {log_rho:.3f}, "
          f"implied analyticity width = {analyticity_width:.2f}")
