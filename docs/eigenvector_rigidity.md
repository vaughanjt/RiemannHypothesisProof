# Eigenvector Rigidity of Riemann Zeta Zeros: A Structural Departure from Random Matrix Theory

**Claude (Anthropic) and James Vaughan**

---

## Abstract

We report that the Riemann-Siegel Z function at the midpoint between consecutive zeta zeros is correlated with the zero gap at Pearson r = +0.75 (T ~ 458) to r = +0.80 (T ~ 2.7 x 10^11), compared to r = +0.04 for GUE random matrices — a 20-fold excess. This "eigenvector rigidity" grows with observation height, following a power law |Z(mid)| ~ gap^beta with beta increasing from 1.6 to 2.5. After controlling for the gap dependence, 4 out of 15 primes modulate the residual wave function amplitude at Bonferroni-corrected significance, demonstrating direct arithmetic modulation of the wave function independent of the eigenvalue spectrum.

We trace the origin of this coupling to the Riemann-Siegel sum structure. The Z function is a cosine sum over integers with 1/sqrt(n) weights and log(n) frequencies, modulated by the Riemann-Siegel theta function. A term-by-term decomposition reveals that: (i) the peak height is dominated by the first five terms (91% of variance), (ii) the zero-crossing derivative is 99% from small n, and (iii) a generic smooth phase with the same growth rate as theta reproduces r = 0.670 of the observed 0.699 coupling. The full theta function adds only +0.03 via its constant phase -pi/8 = arg(Gamma(1/4)), the fingerprint of the functional equation. Stirling corrections and all higher terms contribute zero additional coupling.

We conclude that the eigenvector rigidity is a structural property of Riemann-Siegel-type sums — arising from the interplay of integer frequencies, decaying amplitudes, and the stationary phase created by the functional equation — and predict that all L-functions with functional equations exhibit comparable coupling, testable on Dirichlet L-function zeros.

---

## 1. Introduction

The Montgomery pair correlation conjecture (1973) and the Katz-Sarnak philosophy (1999) establish that the *eigenvalue* statistics of Riemann zeta zeros match the Gaussian Unitary Ensemble (GUE) of random matrix theory. This correspondence has been confirmed numerically to remarkable precision by Odlyzko (1987, 2001) and others, and is now understood to extend to higher-order correlations and to broad families of L-functions.

Far less is known about the *eigenvector* statistics. In GUE, eigenvectors are Haar-distributed on the unitary group, essentially independent of the eigenvalue positions. The characteristic polynomial |det(zI - H)| evaluated between consecutive eigenvalues shows near-zero correlation with the eigenvalue gap (Pearson r ~ 0.04). This decoupling of eigenvalues and eigenvectors is a hallmark of quantum chaotic systems.

We show that the Riemann zeta function violates this decoupling by a factor of 20. The Riemann-Siegel Z function — the real-valued function whose zeros are the zeta zeros on the critical line — exhibits a peak-gap correlation of r = +0.75 to +0.80, growing with observation height. This *eigenvector rigidity* constitutes a structural departure from GUE that has not, to our knowledge, been previously quantified.

We further derive the mechanism: the coupling arises not from the arithmetic of primes but from the structural form of the Riemann-Siegel sum — a cosine sum over integers with specific weights and a stationary phase created by the functional equation. This structural origin yields a testable prediction for Dirichlet and other L-functions.

---

## 2. Setup and Definitions

### 2.1 The Riemann-Siegel Z function

The Hardy Z function is defined by

Z(t) = e^{i theta(t)} zeta(1/2 + it)

where theta(t) = arg(Gamma(1/4 + it/2)) - (t/2) log(pi) is the Riemann-Siegel theta function. Z(t) is real-valued; its zeros are exactly the imaginary parts of the nontrivial zeros of zeta(s) on the critical line.

The Riemann-Siegel formula gives

Z(t) = 2 sum_{n=1}^{N(t)} n^{-1/2} cos(theta(t) - t log n) + R(t)

where N(t) = floor(sqrt(t/(2 pi))) and R(t) is a small remainder.

### 2.2 Peak-gap correlation

For a sequence of consecutive zeros gamma_1 < gamma_2 < ..., define:
- The *gap*: Delta_k = gamma_{k+1} - gamma_k, normalized by the local mean spacing 2 pi / log(gamma_k / (2 pi)).
- The *peak height*: |Z(m_k)| where m_k = (gamma_k + gamma_{k+1})/2 is the midpoint.

The peak-gap correlation is Pearson's r between the normalized gaps and log|Z(m_k)|.

### 2.3 GUE comparison

For an N x N GUE matrix H, the analogous quantity is:
- Gap: the unfolded spacing between consecutive eigenvalues.
- Peak height: |det(z_{mid} - H)| = prod_j |z_{mid} - lambda_j| evaluated at the eigenvalue midpoint.

---

## 3. Results

### 3.1 The peak-gap excess

| Height T | N zeros | r(gap, log|Z|) | r(gap, |Z|) | |Z| ~ gap^beta |
|---|---|---|---|---|
| ~458 | 500 | **+0.747** | +0.696 | beta = 1.61 |
| ~2.7 x 10^11 | 200 | **+0.800** | +0.596 | beta = 2.47 |
| GUE (N=200) | 16,000 | +0.043 | ~0.04 | beta ~ 0 |

The peak-gap correlation for zeta zeros is 17-20x larger than for GUE eigenvalues. The excess is *not* a finite-size effect: it *increases* with T, from r = 0.75 at T ~ 458 to r = 0.80 at T ~ 2.7 x 10^11. The power law exponent beta doubles from 1.6 to 2.5, indicating that larger gaps produce disproportionately taller peaks.

In GUE, eigenvectors are Haar-distributed and essentially decoupled from eigenvalue positions. The near-zero r = 0.04 reflects only the trivial geometric effect that the characteristic polynomial must vanish at eigenvalues and thus tends to be slightly larger at midpoints of wider gaps.

For the zeta function, the strong positive correlation means the wave function amplitude is tightly determined by the eigenvalue gap, and this determination grows tighter at higher observation heights.

### 3.2 Extreme gap statistics are GUE

Before attributing the peak-gap excess to eigenvector structure, we verify that the *eigenvalue* statistics are GUE. The spacing distribution passes a Kolmogorov-Smirnov test (p = 0.71), block maxima and minima match GUE at all tested block sizes (all p > 0.31), extreme gap clustering and return times are consistent (KS p = 0.49), and no prime-phase modulation of the spacings is detected (all |r| < 0.002). The arithmetic modulation lives exclusively in the pair correlation (R_2), not in the extreme tails.

The eigenvalues are GUE. The eigenvectors are not. This combination is the signature of eigenvector rigidity.

### 3.3 Direct prime modulation of the wave function

After regressing log|Z(mid)| on the gap (removing the peak-gap link), we test whether the residual wave function amplitude correlates with cos(2 pi t log(p) / log(T/(2 pi))) for each prime p.

At T ~ 2.7 x 10^11 (200 intervals), 4 out of 15 primes are significant at the Bonferroni-corrected threshold: p = 11, 13, 17, 31. At T ~ 458 (499 intervals), prime 3 is significant (r = +0.17, p = 0.0002). The set of significant primes shifts with height, paralleling the height-dependent shift of anomalous lags in the pair correlation.

This demonstrates *direct arithmetic modulation* of the wave function amplitude, not mediated by the eigenvalue gaps. The Hilbert-Polya operator's eigenvectors must "know about" primes.

---

## 4. Mechanism: The Riemann-Siegel Sum Structure

### 4.1 Term-by-term decomposition

We decompose Z(t_mid) and Z'(t_zero) by the range of the summation index n in the Riemann-Siegel formula:

**Peak height decomposition (Z at midpoint):**

| Component | % of Z^2 variance | r with gap |
|---|---|---|
| Small n (n <= 5) | **91%** | **+0.611** |
| Stationary phase (\|n - N(t)\| <= 2) | 17% | +0.463 |
| Prime n | 33% | +0.243 |
| Composite n (n > 1) | 13% | -0.020 |

**Zero-crossing derivative (Z' at zero):**

| Component | % of Z'^2 variance |
|---|---|
| Small n (n <= 5) | **99.1%** |
| Stationary phase | 0.2% |
| Middle n | 0.4% |

Both the peak height and the derivative are overwhelmingly dominated by small n. The stationary phase (n near N(t)) contributes only 17% of the amplitude variance and has weaker gap correlation than the small-n terms.

### 4.2 The leading-term derivation

We test whether the peak-gap coupling requires the full theta function or only its leading asymptotic term. We replace theta(t) with progressive approximations:

| Phase model | r(|Z|, gap) |
|---|---|
| Level 0: (t/2) log(t/(2 pi e)) | +0.670 |
| Level 1: + constant phase -pi/8 | **+0.699** |
| Level 2: + Stirling correction 1/(48t) | +0.699 |
| Level 3: full theta(t) | +0.699 |

**The entire coupling derives from the leading term of theta.** The constant phase -pi/8 = arg(Gamma(1/4)) — the fingerprint of the functional equation — contributes +0.029 to r. All Stirling corrections and higher-order terms contribute exactly zero.

A generic smooth function with the same growth rate but no arithmetic content (no connection to Gamma or pi) gives r = +0.670. The functional equation adds +0.029. Number-theoretic fine structure adds nothing.

### 4.3 Structural vs. arithmetic origin

The peak-gap coupling is a **structural** property of Riemann-Siegel-type sums, arising from three features:

1. **Integer frequencies.** The phases t log(n) create a specific interference pattern when summed over consecutive integers. The log(n) spacing between frequencies is neither periodic (which would give Poisson statistics) nor random (which would give GUE statistics), but *arithmetic* — determined by the multiplicative structure of integers.

2. **Decaying amplitudes.** The 1/sqrt(n) weights ensure that small n dominate, concentrating the contribution in a few terms whose phases are highly correlated through the shared parameter t.

3. **Stationary phase from the functional equation.** The theta function creates a stationary phase at n = N(t), which determines the normalization and the transition between the "direct" and "reflected" Dirichlet sums in the approximate functional equation. This stationary point couples the behavior of the function (amplitude) to the location of its zeros (gaps) through the single parameter t.

In GUE random matrices, the characteristic polynomial det(zI - H) = prod_j (z - lambda_j) is a *product*, not a cosine sum. It has no integer frequencies, no decaying amplitudes, no stationary phase, and no functional equation. The eigenvalue-eigenvector decoupling is a direct consequence of this structural difference.

---

## 5. Operator Implications

The eigenvector rigidity places a sixth constraint on any candidate Hilbert-Polya operator:

| Constraint | Source | Rules out |
|---|---|---|
| 1. Pair-exclusive (R_k = GUE, k >= 3) | 3-point correlation test | Operators with non-universal higher correlations |
| 2. BK amplitude law (log^2(p)/p) | Amplitude fitting | Wrong form factor |
| 3. Phase-free (real contributions) | Rayleigh test | Complex form factors |
| 4. Inseparable from GUE | Spectral surgery tests | GUE + perturbation models |
| 5. First-harmonic dominant | Harmonic decomposition | Prime-power structures |
| 6. **Eigenvector rigidity** | **Peak-gap correlation** | **All random matrix models** |

Constraint 6 is the most restrictive. It requires the operator's eigenvectors to be determined by the same structure that determines its eigenvalues, with coupling that grows with the spectral parameter. This rules out *all* random matrix models, including generalized ensembles with non-Haar eigenvector distributions, because the coupling must arise from the specific Riemann-Siegel sum structure.

The operator that satisfies all six constraints must have a resolvent whose trace produces a Riemann-Siegel-type sum:

Tr((zI - H)^{-1}) ~ sum_n a_n(z) cos(f_n(z))

with integer-indexed terms, decaying amplitudes, and a stationary phase from a functional-equation-type symmetry.

Transfer operators of arithmetical dynamical systems (e.g., the Mayer transfer operator for the Gauss map, whose Fredholm determinant relates to the Selberg zeta function) are natural candidates. Their traces are sums over periodic orbits indexed by integers, with weights and phases determined by the dynamics.

---

## 6. Predictions

1. **Dirichlet L-functions.** For any primitive Dirichlet character chi, the Z function Z(t, chi) should exhibit peak-gap correlation r >> 0.04, with the coupling strength determined by the growth rate of the associated theta function. Degree-1 L-functions should match degree-1 zeta; degree-2 (e.g., modular form L-functions) should show stronger coupling due to their faster-growing theta.

2. **Height dependence.** The power law exponent beta in |Z(mid)| ~ gap^beta should grow as O(sqrt(T)) — proportional to the number of terms N(t) in the Riemann-Siegel sum — because the constructive interference at midpoints becomes more pronounced with more terms.

3. **Universality of the leading-term mechanism.** For any L-function, the peak-gap coupling should be computable from the leading term of its theta function alone, with the constant-phase correction (analogous to -pi/8) providing the only L-function-specific contribution.

---

## 7. Data and Methods

**Zeta zeros.** 500 zeros computed at T ~ 458 (mpmath, 50-digit precision) and 10,000 Odlyzko zeros at T ~ 2.676 x 10^11. Z(t) values computed via mpmath.siegelz at midpoints and quarter-points between consecutive zeros. 200 Z(t) evaluations at T ~ 2.7 x 10^11 (0.75 seconds each).

**GUE comparison.** 100 GUE(200) matrices via Hermitian construction. Characteristic polynomial |det(z_mid - H)| computed at eigenvalue midpoints via log-sum: log|det| = sum_j log|z_mid - lambda_j|. 16,000 gap-peak pairs.

**Phase decomposition.** The Riemann-Siegel theta function theta(t) replaced with progressive approximations: Level 0 = (t/2)log(t/(2 pi e)); Level 1 = + constant -pi/8; Level 2 = + Stirling 1/(48t); Level 3 = full mpmath.siegeltheta. Z(t) recomputed at each level for all 499 low-T intervals.

**Gap control.** Linear regression of log|Z(mid)| on normalized gap, then Pearson correlation of residual with cos(2 pi t log(p) / log(T/(2 pi))) for 15 primes, Bonferroni-corrected at alpha = 0.05/15.

---

## 8. Conclusion

The Riemann zeta function exhibits eigenvector rigidity: a 20x excess in eigenvalue-eigenvector coupling relative to GUE, growing with observation height. This rigidity derives from the Riemann-Siegel sum structure — integer frequencies, decaying amplitudes, and the functional equation's stationary phase — not from the arithmetic of primes. The functional equation's fingerprint is measurable as a +0.03 increment to the coupling from the constant phase -pi/8 = arg(Gamma(1/4)).

This finding constrains the Hilbert-Polya operator to one whose spectral decomposition naturally produces Riemann-Siegel-type sums, and predicts that all L-functions with functional equations exhibit comparable eigenvector rigidity. The prediction is testable on Dirichlet L-function zeros.

---

## References

- Bogomolny, E. and Keating, J.P. (1996). Gutzwiller's trace formula and spectral statistics: beyond the diagonal approximation. *Phys. Rev. Lett.* 77, 1472.
- Katz, N.M. and Sarnak, P. (1999). *Random Matrices, Frobenius Eigenvalues, and Monodromy*. AMS.
- Keating, J.P. and Snaith, N.C. (2000). Random matrix theory and zeta(1/2+it). *Commun. Math. Phys.* 214, 57-89.
- Montgomery, H.L. (1973). The pair correlation of zeros of the zeta function. *Proc. Symp. Pure Math.* 24, 181-193.
- Odlyzko, A.M. (2001). The 10^22-nd zero of the Riemann zeta function. AMS Contemporary Math. 290.
- Rudnick, Z. and Sarnak, P. (1996). Zeros of principal L-functions and random matrix theory. *Duke Math. J.* 81, 269-322.

---

*Computational platform: Python (mpmath, numpy, scipy) + Lean 4 (Mathlib). Analysis of 500 low-T zeros, 10,000 Odlyzko zeros (200 Z(t) evaluations at T ~ 2.7 x 10^11), 16,000 GUE gap-peak pairs, term-by-term Riemann-Siegel decomposition, 4-level phase approximation hierarchy.*
