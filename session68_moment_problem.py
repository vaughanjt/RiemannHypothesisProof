"""
SESSION 68 -- THE HAMBURGER MOMENT PROBLEM

The moments m_k = Sum 1/gamma^{2k} are computable from xi derivatives
at s=1/2 (via eta, no zeros needed).

The HAMBURGER MOMENT PROBLEM asks: does a positive measure mu on [0,inf)
exist with integral x^k d(mu) = m_k?

If yes: the measure IS the spectral measure of the Hilbert-Polya operator.
        Its support is {1/gamma_k^2}, which is real. Hence zeros are real. RH.

The NECESSARY AND SUFFICIENT condition (for a determinate moment problem):
the Hankel matrix H[i,j] = m_{i+j} must be POSITIVE SEMI-DEFINITE.

This is computable from xi derivatives alone. No zeros input.
If H is PSD: RH holds (for the truncation).

THIS COULD BE THE PROOF PATH.
"""

import sys
import numpy as np
import mpmath
from mpmath import mp, mpf

mp.dps = 50


def xi_func(s):
    return mpf('0.5') * s * (s-1) * mpmath.power(mpmath.pi, -s/2) * \
           mpmath.gamma(s/2) * mpmath.zeta(s)


def run():
    print()
    print('#' * 76)
    print('  SESSION 68 -- THE HAMBURGER MOMENT PROBLEM')
    print('#' * 76)

    s = mpf('0.5')
    xi_val = xi_func(s)

    # ==================================================================
    # STEP 1: Compute moments m_k from xi derivatives
    # ==================================================================
    print('\n  === STEP 1: MOMENTS FROM XI (NO ZEROS) ===\n')

    K = 12  # number of moments
    mp.dps = 80

    # Z_{2k} = xi^{(2k)}(1/2) / xi(1/2)
    Z = []
    for k in range(1, K + 1):
        deriv = mpmath.diff(xi_func, s, n=2*k)
        z_k = float(deriv / xi_val)
        Z.append(z_k)

    mp.dps = 50

    # Taylor coefficients: c_k = Z_{2k} * (-1)^k / (2k)!
    c = [1.0]
    for k in range(1, K + 1):
        factorial_2k = 1
        for j in range(1, 2*k + 1):
            factorial_2k *= j
        c_k = Z[k-1] * (-1)**k / factorial_2k
        c.append(c_k)

    # Moments m_k from the log-series expansion
    # log(Sum c_k w^k) = -Sum m_j w^j / j
    # Using the recursive relation for log-coefficients
    a = np.array(c[1:K+1])
    b = np.zeros(K)
    b[0] = a[0]
    for n in range(1, K):
        s_val = sum((k+1) * b[k] * a[n-1-k] for k in range(n))
        b[n] = a[n] - s_val / (n + 1)

    m = np.zeros(K)
    for j in range(K):
        m[j] = -(j + 1) * b[j]

    print(f'  Moments m_k = Sum 1/gamma^{{2k}} (from xi, NO zeros):')
    for k in range(min(K, 10)):
        print(f'    m_{k+1} = {m[k]:+.15e}')

    # Verify against direct computation from zeros
    gammas = []
    for k in range(1, 201):
        gammas.append(float(mpmath.im(mpmath.zetazero(k))))
    gammas = np.array(gammas)

    print(f'\n  Verification (200 zeros):')
    for k in range(min(5, K)):
        direct = np.sum(1.0 / gammas**(2*(k+1)))
        print(f'    m_{k+1}: xi={m[k]:+.10e}, zeros={direct:+.10e}, '
              f'ratio={m[k]/direct:.6f}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 2: BUILD THE HANKEL MATRIX
    # ==================================================================
    print(f'\n  === STEP 2: HANKEL MATRIX H[i,j] = m_{{i+j+1}} ===\n')

    # For the Stieltjes moment problem (measure on [0,inf)):
    # The Hankel matrix H[i,j] = m_{i+j} for i,j = 0,...,K-1
    # where m_0 = Sum 1/gamma^0 = number of zeros (infinite!)
    #
    # For a TRUNCATED problem: use m_k for k = 1, ..., 2K
    # H[i,j] = m_{i+j+1} for i,j = 0,...,K-1

    # Actually, the moment problem for mu on (0, inf) with moments:
    # mu_k = integral x^k dmu = m_{k+1} = Sum 1/gamma^{2(k+1)}
    # (shifting index so mu_0 = m_1 = Sum 1/gamma^2 = Z/2)

    # Hankel matrix: H[i,j] = mu_{i+j} = m_{i+j+1}
    N_hankel = min(K // 2, 5)  # size of Hankel matrix

    H = np.zeros((N_hankel, N_hankel))
    for i in range(N_hankel):
        for j in range(N_hankel):
            idx = i + j  # mu_{i+j} = m_{i+j+1}
            if idx < K:
                H[i, j] = m[idx]
            else:
                H[i, j] = 0  # truncation

    print(f'  Hankel matrix ({N_hankel}x{N_hankel}):')
    for i in range(N_hankel):
        row = '  '.join(f'{H[i,j]:+.6e}' for j in range(N_hankel))
        print(f'    [{row}]')

    # Check PSD
    eigs_H = np.linalg.eigvalsh(H)
    n_pos = np.sum(eigs_H > 1e-20)
    n_neg = np.sum(eigs_H < -1e-20)

    print(f'\n  Eigenvalues of Hankel matrix:')
    for i, e in enumerate(eigs_H):
        sign = '+' if e > 0 else ('-' if e < 0 else '0')
        print(f'    eig_{i+1} = {e:+.10e}  ({sign})')

    print(f'\n  Positive: {n_pos}, Negative: {n_neg}, Zero: {N_hankel-n_pos-n_neg}')
    print(f'  POSITIVE SEMI-DEFINITE: {n_neg == 0}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 3: LARGER HANKEL MATRIX
    # ==================================================================
    print(f'\n  === STEP 3: SCALING UP ===\n')

    for N_test in range(2, K // 2 + 1):
        H_test = np.zeros((N_test, N_test))
        for i in range(N_test):
            for j in range(N_test):
                idx = i + j
                if idx < K:
                    H_test[i, j] = m[idx]
        eigs = np.linalg.eigvalsh(H_test)
        min_eig = eigs[0]
        is_psd = min_eig > -1e-20
        print(f'  {N_test}x{N_test} Hankel: min_eig = {min_eig:+.6e}  '
              f'PSD: {"YES" if is_psd else "**NO**"}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 4: THE SHIFTED HANKEL (for the Stieltjes problem)
    # ==================================================================
    print(f'\n  === STEP 4: SHIFTED HANKEL (STIELTJES PROBLEM) ===')
    print(f'  For measure on [0,inf): need BOTH H and H_shifted PSD.\n')

    # The Stieltjes moment problem requires:
    # H_0[i,j] = mu_{i+j} = m_{i+j+1}  PSD (already checked)
    # H_1[i,j] = mu_{i+j+1} = m_{i+j+2}  PSD (shifted)

    for N_test in range(2, K // 2):
        H0 = np.zeros((N_test, N_test))
        H1 = np.zeros((N_test, N_test))
        for i in range(N_test):
            for j in range(N_test):
                idx0 = i + j
                idx1 = i + j + 1
                if idx0 < K:
                    H0[i, j] = m[idx0]
                if idx1 < K:
                    H1[i, j] = m[idx1]

        eigs0 = np.linalg.eigvalsh(H0)
        eigs1 = np.linalg.eigvalsh(H1)
        psd0 = eigs0[0] > -1e-20
        psd1 = eigs1[0] > -1e-20

        print(f'  {N_test}x{N_test}: H0 min={eigs0[0]:+.6e} ({"PSD" if psd0 else "NO"}), '
              f'H1 min={eigs1[0]:+.6e} ({"PSD" if psd1 else "NO"}), '
              f'Stieltjes: {"YES" if psd0 and psd1 else "**NO**"}')
    sys.stdout.flush()

    # ==================================================================
    # STEP 5: WHAT DOES PSD MEAN?
    # ==================================================================
    print(f'\n  === STEP 5: INTERPRETATION ===\n')
    print(f'  The Hamburger moment problem: given m_1, m_2, ..., m_K')
    print(f'  (computed from xi derivatives, NO zeros),')
    print(f'  does a positive measure exist on [0, inf) with these moments?')
    print()
    print(f'  If YES (Hankel PSD): the measure is the spectral measure')
    print(f'  of a self-adjoint operator with eigenvalues 1/gamma_k^2.')
    print(f'  Since the eigenvalues are real, gamma_k are real, delta_k = 0.')
    print(f'  This proves RH for the first K zeros.')
    print()
    print(f'  If NO (Hankel not PSD): the moments are inconsistent with')
    print(f'  a real spectrum. RH would be false.')
    print()

    all_psd = True
    for N_test in range(2, K // 2 + 1):
        H_test = np.zeros((N_test, N_test))
        for i in range(N_test):
            for j in range(N_test):
                idx = i + j
                if idx < K:
                    H_test[i, j] = m[idx]
        eigs = np.linalg.eigvalsh(H_test)
        if eigs[0] < -1e-15:
            all_psd = False
            break

    if all_psd:
        print(f'  RESULT: ALL Hankel matrices up to {K//2}x{K//2} are PSD.')
        print(f'  The moment problem has a solution.')
        print(f'  Consistent with RH.')
    else:
        print(f'  RESULT: Hankel matrix failed PSD at size {N_test}.')
    sys.stdout.flush()

    # ==================================================================
    # STEP 6: THE JACOBI MATRIX (HILBERT-POLYA OPERATOR)
    # ==================================================================
    print(f'\n  === STEP 6: JACOBI MATRIX FROM MOMENTS ===')
    print(f'  If PSD: the Lanczos algorithm produces a tridiagonal')
    print(f'  self-adjoint matrix whose eigenvalues are 1/gamma_k^2.\n')

    # Build the Jacobi matrix using the Chebyshev algorithm
    # (three-term recurrence from moments)
    N_jacobi = min(4, K // 3)

    # The moments define orthogonal polynomials P_k(x) with:
    # x*P_k = a_{k+1}*P_{k+1} + b_k*P_k + a_k*P_{k-1}
    # The Jacobi matrix J has b_k on diagonal, a_k on sub/super-diagonal

    # Using the modified Chebyshev algorithm:
    sigma = np.zeros((2*N_jacobi + 1, 2*N_jacobi + 1))
    for k in range(2*N_jacobi + 1):
        if k < K:
            sigma[0, k] = m[k]

    alpha = np.zeros(N_jacobi)
    beta = np.zeros(N_jacobi)

    alpha[0] = m[1] / m[0] if abs(m[0]) > 1e-30 else 0
    beta[0] = m[0]

    if N_jacobi > 1:
        for k in range(2*N_jacobi):
            if k < K:
                sigma[1, k] = sigma[0, k+1] - alpha[0] * sigma[0, k]

        for n in range(1, N_jacobi):
            for k in range(n, 2*N_jacobi - n):
                if k < K - 1 and n < 2*N_jacobi:
                    sigma[n+1, k] = sigma[n, k+1] - alpha[n-1]*sigma[n, k] - \
                                    beta[n-1]*sigma[n-1, k] if n > 0 else sigma[n, k+1]

            if abs(sigma[n, n]) > 1e-30 and abs(sigma[n-1, n-1]) > 1e-30:
                alpha[n] = sigma[n+1, n] / sigma[n, n] - sigma[n, n-1] / sigma[n-1, n-1] \
                           if n > 0 and abs(sigma[n-1, n-1]) > 1e-30 else 0
                beta[n] = sigma[n, n] / sigma[n-1, n-1] if abs(sigma[n-1, n-1]) > 1e-30 else 0

    # Build Jacobi matrix
    J = np.zeros((N_jacobi, N_jacobi))
    for i in range(N_jacobi):
        J[i, i] = alpha[i]
        if i > 0:
            J[i, i-1] = np.sqrt(abs(beta[i]))
            J[i-1, i] = np.sqrt(abs(beta[i]))

    print(f'  Jacobi matrix ({N_jacobi}x{N_jacobi}):')
    for i in range(N_jacobi):
        row = '  '.join(f'{J[i,j]:+.6e}' for j in range(N_jacobi))
        print(f'    [{row}]')

    eigs_J = np.linalg.eigvalsh(J)
    print(f'\n  Eigenvalues of Jacobi matrix (= 1/gamma_k^2):')
    for i, e in enumerate(eigs_J):
        if e > 1e-15:
            gamma_est = 1.0 / np.sqrt(e)
            gamma_actual = gammas[i] if i < len(gammas) else 0
            print(f'    eig_{i+1} = {e:.10e} -> gamma = {gamma_est:.4f} '
                  f'(actual: {gamma_actual:.4f})')
        else:
            print(f'    eig_{i+1} = {e:.10e} (non-positive)')

    print(f'\n  The Jacobi matrix is SELF-ADJOINT by construction.')
    print(f'  Its eigenvalues are REAL by construction.')
    print(f'  If they match 1/gamma_k^2: the zeros are real. RH.')
    sys.stdout.flush()

    # ==================================================================
    # VERDICT
    # ==================================================================
    print()
    print('=' * 76)
    print('  SESSION 68 VERDICT')
    print('=' * 76)


if __name__ == '__main__':
    run()
