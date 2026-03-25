"""EXACT FORMULA FOR b_chi(k).

b_chi(k) = sum_{n=1}^inf chi_4(n) * (n mod k) / k / (n(n+1))

Decompose using (n mod k) = n - k*floor(n/k):
  b_chi(k) = (1/k) sum_n chi_4(n) * n / (n(n+1)) - sum_n chi_4(n) * floor(n/k) / (n(n+1))
           = (1/k) sum_n chi_4(n) / (n+1) - sum_n chi_4(n) * floor(n/k) / (n(n+1))

The first sum is EXACTLY: (1/k) * L_shifted where
  L_shifted = sum chi_4(n)/(n+1) = pi/4 - sum chi_4(n)/(n(n+1))

The second sum involves floor(n/k), which counts complete periods.
  floor(n/k) = 0 for n < k, 1 for k <= n < 2k, etc.

  sum_n chi_4(n) * floor(n/k) / (n(n+1))
  = sum_{q=1}^inf sum_{n=qk}^{(q+1)k-1} chi_4(n) * q / (n(n+1))
    [contribution from each period q]
  + terms where floor(n/k) increases

Actually simpler: floor(n/k) = sum_{j=1}^inf 1_{n >= jk}

So: sum_n chi_4(n) * floor(n/k) / (n(n+1)) = sum_{j=1}^inf sum_{n=jk}^inf chi_4(n)/(n(n+1))

Define: T(m) = sum_{n=m}^inf chi_4(n) / (n(n+1))  (tail sum starting at m)

Then: sum_n chi_4(n) * floor(n/k) / (n(n+1)) = sum_{j=1}^inf T(jk)

And: b_chi(k) = L_shifted/k - sum_{j=1}^inf T(jk)

This is an EXACT identity! Let's verify and analyze T(m).
"""
import numpy as np

M = 200000

def chi_4(n):
    r = n % 4
    if r == 1: return 1.0
    if r == 3: return -1.0
    return 0.0

chi_arr = np.array([chi_4(n) for n in range(1, M+1)])
w_arr = 1.0 / (np.arange(1, M+1) * np.arange(2, M+2))

# ============================================================
# STEP 1: Compute and analyze T(m)
# ============================================================
print("="*70)
print("STEP 1: TAIL SUM T(m) = sum_{n>=m} chi_4(n)/(n(n+1))")
print("="*70)

# Compute T(m) as reverse cumulative sum
terms = chi_arr * w_arr
T_arr = np.cumsum(terms[::-1])[::-1]  # T(m) = sum from m to M

L_full = T_arr[0]  # sum from n=1 to M
L_shifted = np.sum(chi_arr / np.arange(2, M+2))

print(f"\n  L_full = sum chi_4(n)/(n(n+1)) = {L_full:.10f}")
print(f"  L_shifted = sum chi_4(n)/(n+1) = {L_shifted:.10f}")
print(f"  pi/4 = {np.pi/4:.10f}")
print(f"  L_shifted + L_full = {L_shifted + L_full:.10f} (should = pi/4)")

# T(m) values
print(f"\n  {'m':>6} {'T(m)':>14} {'|T(m)|':>12} {'m*|T(m)|':>12}")
print(f"  {'-'*46}")
for m in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]:
    if m <= M:
        Tm = T_arr[m-1]
        print(f"  {m:>6} {Tm:>14.8e} {abs(Tm):>12.8e} {m*abs(Tm):>12.8f}")

# T(m) should be O(1/m) because chi_4(n)/(n(n+1)) is an alternating series
# after removing the smooth part.

# More precisely: T(m) = sum_{n>=m} chi_4(n)/(n(n+1))
# For m = 4q+r: the sum starts at a specific residue class
# By partial summation: T(m) ~ chi_4(m)/(m*(m+1)) + ...
# The alternating nature gives T(m) ~ O(1/m)

# Fit the decay
ms = np.arange(10, 5000)
Ts = T_arr[ms - 1]
mask = np.abs(Ts) > 1e-15
coeffs_T = np.polyfit(np.log(ms[mask]), np.log(np.abs(Ts[mask])), 1)
print(f"\n  |T(m)| ~ m^{{{coeffs_T[0]:.4f}}} (should be ~ -1)")

# Pattern of T(m) by m mod 4
print(f"\n  T(m) pattern by m mod 4 (m=100..120):")
for m in range(100, 121):
    Tm = T_arr[m-1]
    print(f"    m={m:>3} ({m%4}mod4): T(m) = {Tm:>12.8e}", end="")
    if m % 4 == 1:
        print(f"  (expected: +1/(m(m+1)) area)")
    elif m % 4 == 3:
        print(f"  (expected: -1/(m(m+1)) area)")
    else:
        print(f"  (expected: ~0, chi_4=0)")


# ============================================================
# STEP 2: Verify the identity b_chi(k) = L_shifted/k - sum T(jk)
# ============================================================
print("\n" + "="*70)
print("STEP 2: VERIFY b_chi(k) = L_shifted/k - sum_{j>=1} T(jk)")
print("="*70)

print(f"\n  {'k':>5} {'b_chi actual':>14} {'L_sh/k - sum T':>15} {'error':>12}")
print(f"  {'-'*48}")

for k in range(2, 51):
    # Actual b_chi(k)
    b_actual = np.sum(chi_arr * ((np.arange(1, M+1) % k) / k) * w_arr)

    # Formula: L_shifted/k - sum_{j=1}^{M/k} T(jk)
    tail_sum = sum(T_arr[j*k - 1] for j in range(1, M//k + 1) if j*k <= M)
    b_formula = L_shifted / k - tail_sum

    error = abs(b_actual - b_formula)

    if k <= 20 or k % 10 == 0:
        print(f"  {k:>5} {b_actual:>14.10f} {b_formula:>15.10f} {error:>12.2e}")

# The identity holds! Now analyze sum T(jk)
print(f"\n  The identity is EXACT (up to truncation).")


# ============================================================
# STEP 3: Analyze sum_{j>=1} T(jk) — what does it look like?
# ============================================================
print("\n" + "="*70)
print("STEP 3: STRUCTURE OF sum_{j>=1} T(jk)")
print("="*70)

print(f"\n  {'k':>5} {'sum T(jk)':>14} {'L_sh/k':>12} {'b_chi':>12} {'sum_T/b_chi':>12}")
print(f"  {'-'*58}")

for k in range(2, 51):
    tail_sum = sum(T_arr[j*k - 1] for j in range(1, M//k + 1) if j*k <= M)
    Lk = L_shifted / k
    b_k = Lk - tail_sum

    if k <= 20 or k % 10 == 0:
        ratio = tail_sum / (b_k + 1e-30)
        print(f"  {k:>5} {tail_sum:>14.8e} {Lk:>12.8e} {b_k:>12.8e} {ratio:>12.4f}")

# The tail sum is LARGE — it nearly cancels L_shifted/k!
# b_chi(k) is the SMALL DIFFERENCE between L_shifted/k and sum T(jk).

# Both are O(1/k), and their difference is also O(1/k) but with a smaller constant.

# KEY: sum T(jk) = sum_{j>=1} T(jk)
# T(jk) ~ C_{jk} / (jk) where C depends on jk mod 4.
# So sum T(jk) ~ (1/k) sum_{j>=1} C_{jk}/j

# For k even: jk mod 4 depends on j mod 2 (if k=2mod4) or is always 0 (if k=0mod4)
# For k = 0 mod 4: jk = 0 mod 4 always, so T(jk) follows one pattern
# For k = 2 mod 4: jk mod 4 alternates between 2 and 0

# sum_{j>=1} C_{jk}/j is a Dirichlet-series-like sum
# that can be evaluated in terms of L(1, chi_4) = pi/4.

# Let me check: for k = 4, sum T(4j) should be computable.
print(f"\n  For k=4: sum T(4j) for j=1,2,3,...")
print(f"    T(4) = {T_arr[3]:.8e}")
print(f"    T(8) = {T_arr[7]:.8e}")
print(f"    T(12) = {T_arr[11]:.8e}")
print(f"    T(16) = {T_arr[15]:.8e}")
# These should all have the same sign pattern (4j = 0 mod 4)


# ============================================================
# STEP 4: THE EXACT FORMULA
# ============================================================
print("\n" + "="*70)
print("STEP 4: EXACT FORMULA ATTEMPT")
print("="*70)

# b_chi(k) = L_shifted/k - sum_{j=1}^inf T(jk)
#
# Write T(m) = sum_{n>=m} chi_4(n)/(n(n+1)) = sum_{n>=m} chi_4(n)*(1/n - 1/(n+1))
#
# By partial summation: T(m) = S(m-1)/m + sum_{n>=m} S(n)/(n(n+1)*(?)...)
# where S(n) = sum_{j=1}^n chi_4(j) is the partial sum of chi_4.
#
# S(n) = #{j<=n: j=1mod4} - #{j<=n: j=3mod4}
# For n = 4q+r:
#   S(4q) = 0 (equal number of +1 and -1)
#   S(4q+1) = 1
#   S(4q+2) = 1
#   S(4q+3) = 0
#
# So S(n) is PERIODIC with period 4, values {1, 1, 0, 0} (starting from n=1).
# Actually: S(1)=1, S(2)=1, S(3)=0, S(4)=0, S(5)=1, S(6)=1, S(7)=0, S(8)=0,...

S_chi4 = np.cumsum(chi_arr)
print(f"  Partial sums S(n) = sum chi_4(j) for j=1..n:")
for n in range(1, 17):
    print(f"    S({n}) = {S_chi4[n-1]:.0f}", end="  ")
    if n % 4 == 0: print()

print(f"\n  S(n) is PERIODIC with period 4: pattern [1, 1, 0, 0]")
print(f"  |S(n)| <= 1 for all n")

# With S(n) bounded, partial summation gives:
# T(m) = -S(m-1)/m + sum_{n>=m} S(n) * [1/(n(n+1)) - 1/((n+1)(n+2))]
#       ... this simplifies using S bounded.
#
# More directly: since S(n) is periodic with values in {0, 1}:
# T(m) = sum_{n>=m} chi_4(n)/(n(n+1))
# = sum_{n>=m, n=1mod4} 1/(n(n+1)) - sum_{n>=m, n=3mod4} 1/(n(n+1))
#
# Each sub-sum is a sum over an arithmetic progression:
# sum_{n=r mod 4, n>=m} 1/(n(n+1)) = (1/4) sum related to digamma
#
# Specifically: sum_{n=0}^inf 1/((4n+r)(4n+r+1)) = [psi((r+1)/4) - psi(r/4)] / 4
# where psi is the digamma function.
#
# But for the TAIL starting at m: we need the partial sum.

# For our PROOF, the key insight is simpler:
# |T(m)| <= 1/m (since S is bounded by 1 and we can use Abel summation)
# And sum_{j>=1} T(jk) = O(1/k * sum 1/j) ... wait, that diverges.
# But the SIGNS of T(jk) alternate, so the sum converges.

# Let's check: does sum T(jk) converge absolutely?
print(f"\n  Convergence of sum |T(jk)| for k=4:")
partial = 0
for j in range(1, 1001):
    if j*4 <= M:
        partial += abs(T_arr[j*4-1])
print(f"    sum |T(4j)| for j=1..1000 = {partial:.8f}")
print(f"    (For comparison: L_shifted/4 = {L_shifted/4:.8f})")

# Does it converge?
partial_100 = sum(abs(T_arr[j*4-1]) for j in range(1, 101) if j*4 <= M)
partial_1000 = sum(abs(T_arr[j*4-1]) for j in range(1, 1001) if j*4 <= M)
print(f"    sum |T(4j)| to j=100: {partial_100:.8f}")
print(f"    sum |T(4j)| to j=1000: {partial_1000:.8f}")
print(f"    Ratio: {partial_1000/partial_100:.4f} (would be 1.0 if converged)")

# The absolute sum may diverge as log, but the SIGNED sum converges.
# This is because T(jk) alternates in a pattern related to chi_4.

# For the PROOF, what matters is:
# b_chi(k) = L_shifted/k - sum T(jk) = explicit_value/k
# where the explicit value is computable from the C(k) identity.

# Let's verify the CLEANEST form:
# b_chi(k) = (1/k) * [L_shifted - k * sum_{j>=1} T(jk)]

print(f"\n  Testing: k * sum T(jk) as a function of k:")
print(f"  {'k':>5} {'k*sum_T(jk)':>14} {'L_shifted':>12} {'L_sh - k*sum_T':>16}")
print(f"  {'-'*50}")

for k in range(2, 31):
    ksum = k * sum(T_arr[j*k-1] for j in range(1, M//k + 1) if j*k <= M)
    diff = L_shifted - ksum
    print(f"  {k:>5} {ksum:>14.8f} {L_shifted:>12.8f} {diff:>16.8f}")

# diff = k * b_chi(k). Let's see if THIS has a pattern.
print(f"\n  k * b_chi(k) values:")
print(f"  {'k':>5} {'k*b_chi':>12} {'k mod 4':>8}")
print(f"  {'-'*24}")
for k in range(2, 31):
    b_k = np.sum(chi_arr * ((np.arange(1, M+1) % k) / k) * w_arr)
    print(f"  {k:>5} {k*b_k:>12.8f} {k%4:>8}")
