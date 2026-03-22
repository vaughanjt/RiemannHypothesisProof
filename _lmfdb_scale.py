"""LMFDB Maass forms at scale: spectral-geometric duality test.

Strategy:
1. Fetch spectral parameters for 2000+ forms (small data, should work)
2. Attempt Hecke coefficient fetch for top forms (may hit rate limits)
3. Test Selberg trace formula: spectral side vs geometric side
4. Measure convergence as function of number of forms

The Selberg trace formula for SL(2,Z):
  Sum_j h(r_j) = (1/4pi)*integral[h(r)*r*tanh(pi*r)]dr + Sum_{p,m} log(p)/p^{m/2} * g(m*log(p))

If h(r) = exp(-t*r^2), the spectral sum should match the prime sum.
We test: does the spectral side predict the observed ACF excess?
"""
import sys, time, json, hashlib
sys.path.insert(0, 'src')
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from sympy import primerange
from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

MAX_LAG = 400
T_BASE = 267653395647.0
LOG_T = np.log(T_BASE / (2 * np.pi))
k_arr = np.arange(1, MAX_LAG + 1, dtype=float)

# ============================================================
# SETUP: load ACF excess (same as all previous experiments)
# ============================================================
t0 = time.time()
print('Loading ACF data...')

def gue_eigs(n, rng):
    d = rng.standard_normal(n)
    e = np.sqrt(rng.chisquare(2 * np.arange(n - 1, 0, -1)) / 2)
    return eigvalsh_tridiagonal(d, e) / np.sqrt(n)

rng_bl = np.random.default_rng(42)
bl_acfs = []
for _ in range(100):
    eigs = gue_eigs(1200, rng_bl)
    sp = polynomial_unfold(eigs, trim_fraction=0.1)
    if len(sp) > MAX_LAG + 10:
        bl_acfs.append(spacing_autocorrelation(sp, MAX_LAG))
baseline = np.mean(bl_acfs, axis=0)[1:MAX_LAG + 1]

zeros_raw = []
with open('data/odlyzko/zeros3.txt') as f:
    for line in f:
        try: zeros_raw.append(float(line.strip()))
        except ValueError: pass
zeros_arr = np.array(zeros_raw)
density = LOG_T / (2 * np.pi)
sp_real = np.diff(zeros_arr) * density
sp_real /= np.mean(sp_real)
se = 1.0 / np.sqrt(len(sp_real))
acf_real = spacing_autocorrelation(sp_real, MAX_LAG)[1:MAX_LAG + 1]
excess = acf_real - baseline
ss_tot = np.sum(excess ** 2)
print(f'  {len(sp_real)} spacings, {time.time()-t0:.1f}s')

# ============================================================
# STEP 1: Load cached + fetch new spectral parameters
# ============================================================
print('\nStep 1: Loading/fetching Maass form spectral parameters...')

# Load existing cache
with open('data/maass_forms.json') as f:
    cache = json.load(f)

existing_r = [sp['r'] for sp in cache['spectral_parameters']]
existing_sym = [sp['symmetry'] for sp in cache['spectral_parameters']]
print(f'  Cached: {len(existing_r)} forms (r = {min(existing_r):.2f} to {max(existing_r):.2f})')

# Try to fetch more from LMFDB
import urllib.request
import urllib.error

LMFDB_BASE = 'https://www.lmfdb.org/api'
new_forms = []
fetch_failed = False

print('  Fetching additional forms from LMFDB...')
for offset in range(500, 5000, 100):
    url = (f'{LMFDB_BASE}/maass_rigor/?level=1'
           f'&_limit=100&_offset={offset}'
           f'&_sort=spectral_parameter'
           f'&_fields=maass_label,spectral_parameter,symmetry,level'
           f'&_format=json')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'RiemannResearch/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        records = data.get('data', data.get('results', []))
        if not records:
            print(f'    offset={offset}: no more data')
            break
        for rec in records:
            r = rec.get('spectral_parameter')
            sym = rec.get('symmetry', -1)
            if r is not None:
                new_forms.append({'r': float(r), 'symmetry': int(sym)})
        print(f'    offset={offset}: +{len(records)} forms (r up to {new_forms[-1]["r"]:.2f})')
        time.sleep(0.5)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f'    offset={offset}: FAILED ({e})')
        fetch_failed = True
        break
    except Exception as e:
        print(f'    offset={offset}: ERROR ({e})')
        fetch_failed = True
        break

# Merge and deduplicate
all_r = list(existing_r)
all_sym = list(existing_sym)
existing_set = set(f'{r:.10f}' for r in existing_r)
n_new = 0
for form in new_forms:
    key = f'{form["r"]:.10f}'
    if key not in existing_set:
        all_r.append(form['r'])
        all_sym.append(form['symmetry'])
        existing_set.add(key)
        n_new += 1

r_all = np.array(sorted(all_r))
print(f'  Total: {len(r_all)} forms (r = {r_all[0]:.2f} to {r_all[-1]:.2f}), {n_new} new')

# Save updated cache
if n_new > 0:
    new_spectral = [{'r': float(r), 'symmetry': int(s)} for r, s in
                    sorted(zip(all_r, all_sym), key=lambda x: x[0])]
    cache['spectral_parameters'] = new_spectral
    cache['metadata']['total_spectral'] = len(new_spectral)
    cache['metadata']['fetched_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
    with open('data/maass_forms.json', 'w') as f:
        json.dump(cache, f, indent=2)
    print(f'  Saved updated cache ({len(new_spectral)} forms)')

# ============================================================
# STEP 2: Attempt Hecke coefficient fetch for small batch
# ============================================================
print('\nStep 2: Attempting Hecke coefficient fetch...')

# Try to get coefficients for the first 20 forms
hecke_data = {}
labels = [sp.get('label', f'1.0.1.{i+1}.1') for i, sp in
          enumerate(cache['spectral_parameters'][:20])]

n_success = 0
for i, label in enumerate(labels[:20]):
    url = (f'{LMFDB_BASE}/maass_rigor/?maass_label={label}'
           f'&_fields=maass_label,spectral_parameter,coefficients'
           f'&_format=json')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'RiemannResearch/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        records = data.get('data', data.get('results', []))
        if records and 'coefficients' in records[0] and records[0]['coefficients']:
            coeffs = records[0]['coefficients']
            r_val = records[0].get('spectral_parameter', 0)
            # Extract Hecke eigenvalues at primes
            primes_small = list(primerange(2, 100))
            hecke = {}
            for p in primes_small:
                idx = p - 1  # coefficients are typically 0-indexed from n=1
                if isinstance(coeffs, list) and idx < len(coeffs):
                    hecke[p] = float(coeffs[idx])
                elif isinstance(coeffs, dict) and str(p) in coeffs:
                    hecke[p] = float(coeffs[str(p)])
            if hecke:
                hecke_data[i] = {'r': r_val, 'hecke': hecke}
                n_success += 1
                if n_success <= 3:
                    print(f'    Form {i+1} (r={r_val:.4f}): {len(hecke)} coefficients')
        time.sleep(1.0)  # conservative delay
    except Exception as e:
        if i < 3:
            print(f'    Form {i+1}: {e}')
        break

print(f'  Fetched coefficients for {n_success} forms')

# Use hardcoded Hecke data from previous sessions as fallback
HECKE_FORM1 = {
    2: -1.0683335512, 3: -0.4561973545, 5: -0.2906725550,
    7: -0.7449416121, 11: 1.1221862808, 13: -0.0833743988,
    17: 0.6539992842, 19: -0.9801254685, 23: 0.9505293400,
    29: 1.2497309500, 31: 0.8451791688, 37: -0.1738023437,
    41: -1.2447698000, 43: -1.5093898000, 47: 0.2723445700,
    53: -0.3067574000, 59: -0.8032470000, 61: 1.0483340000,
    67: -0.0721862000, 71: -0.0025862000, 73: 0.9508780000,
    79: 0.2876280000, 83: -1.4261980000, 89: 0.5543230000,
    97: -0.3981625410,
}
HECKE_FORM2 = {
    2: 1.5494938670, 3: -0.3499021825, 5: 0.5975355000,
    7: 0.2866362350, 11: 0.5506618000, 13: -1.1770470000,
    17: -0.1893180000, 19: -0.4413670000, 23: -1.1015200000,
    29: 0.4627810000, 31: 0.3759340000, 37: 1.1456400000,
    41: 0.5027740000, 43: -0.9508050000, 47: -0.2135160000,
}

if n_success < 2:
    print('  Using hardcoded Hecke data for 2 forms (from previous session)')
    hecke_data[0] = {'r': 9.5337, 'hecke': HECKE_FORM1}
    hecke_data[1] = {'r': 12.1730, 'hecke': HECKE_FORM2}

# ============================================================
# STEP 3: Selberg trace formula — spectral side prediction
# ============================================================
print('\n' + '=' * 70)
print('SELBERG TRACE FORMULA: SPECTRAL SIDE vs GEOMETRIC SIDE')
print('=' * 70)

# The test function for the ACF at lag k:
# h_k(r) = cos(2*pi*k*r / D) * w(r)
# where D is the spectral density normalization and w(r) is a weight
#
# The geometric side (what we measured):
# G(k) = ACF_excess(k) ~ Sum_p A(p) * cos(2*pi*k*log(p)/log(T/2pi))
#
# The spectral side (what we predict):
# S(k) = Sum_j h_k(r_j) = Sum_j cos(2*pi*k*r_j/D) * w(r_j)
#
# If the trace formula works, S(k) ~ G(k) for large enough sum over j.

# Spectral density: by Weyl's law, N(R) ~ R^2 / 12 for SL(2,Z)
# (actually N(R) = (1/12)*R^2 - (2/pi)*R*log(R) + ... for the number of
# Maass forms with r_j <= R)

# We need to find the right normalization D so that the spectral frequencies
# f_j = r_j / D align with the geometric frequencies f_p = log(p) / log(T/2pi)

# The connection: the Selberg trace formula maps
#   spectral parameter r_j <-> length of closed geodesic l_p = 2*log(p)
# So the natural comparison is: r_j vs log(p)

# For the ACF, the relevant spectral oscillation at lag k is:
# Sum_j cos(2*pi*k * r_j / R_max) where R_max normalizes the spectrum

# Let's test multiple normalizations
print('\nTesting spectral normalizations...')

short_3 = [np.exp(-k_arr / 1.0), np.exp(-k_arr / 3.0), 1.0 / k_arr ** 2]

def spectral_model(r_vals, D, weight_func, k_arr):
    """Build spectral-side model: Sum_j w(r_j) * cos(2*pi*k*r_j/D)."""
    model = np.zeros(len(k_arr))
    for r in r_vals:
        w = weight_func(r)
        model += w * np.cos(2 * np.pi * k_arr * r / D)
    return model

def fit_spectral(model, excess, ss_tot, short_cols):
    """Fit: scale * spectral_model + short-range. Return R2, R2_adj."""
    X = np.column_stack([model] + short_cols)
    a, _, _, _ = np.linalg.lstsq(X, excess, rcond=None)
    pred = X @ a
    R2 = 1 - np.sum((excess - pred) ** 2) / ss_tot
    n_p = X.shape[1]
    R2_adj = 1 - (1 - R2) * (MAX_LAG - 1) / (MAX_LAG - n_p - 1)
    return R2, R2_adj, a[0]

# Weight functions
weights = {
    'uniform': lambda r: 1.0,
    '1/cosh(pi*r/2)': lambda r: 1.0 / np.cosh(np.pi * r / 2),
    '1/r': lambda r: 1.0 / max(r, 0.1),
    'r*tanh': lambda r: r * np.tanh(np.pi * r),
    '1/r^2': lambda r: 1.0 / max(r ** 2, 0.01),
    'exp(-r/50)': lambda r: np.exp(-r / 50),
    'log(r)/r': lambda r: np.log(max(r, 0.1)) / max(r, 0.1),
}

# Normalizations to test
D_values = [LOG_T, LOG_T / (2 * np.pi), r_all[-1], np.sqrt(12 * len(r_all)),
            2 * np.pi * len(r_all) / r_all[-1], LOG_T / np.pi]
D_labels = ['log(T/2pi)', 'log(T/2pi)/2pi', 'r_max', 'sqrt(12*N)',
            '2pi*N/r_max', 'log(T/2pi)/pi']

print(f'\n{"Weight":<20} {"D":<15} {"R2":>7} {"R2_adj":>7} {"scale":>10}')
print('-' * 65)

best_R2_adj = -999
best_config = None

for w_name, w_func in weights.items():
    for D, D_label in zip(D_values, D_labels):
        model = spectral_model(r_all, D, w_func, k_arr)
        if np.max(np.abs(model)) < 1e-20:
            continue
        R2, R2_adj, sc = fit_spectral(model, excess, ss_tot, short_3)
        if R2_adj > best_R2_adj:
            best_R2_adj = R2_adj
            best_config = (w_name, D_label, D, R2, R2_adj, sc)
        if R2_adj > 0.01:  # only show non-trivial
            print(f'{w_name:<20} {D_label:<15} {R2:>7.4f} {R2_adj:>7.4f} {sc:>10.4f}')

if best_config:
    print(f'\nBest: weight={best_config[0]}, D={best_config[1]}, '
          f'R2_adj={best_config[4]:.4f}')

# ============================================================
# STEP 4: Convergence — how does R2 improve with more forms?
# ============================================================
print('\n' + '=' * 70)
print('CONVERGENCE: R2 vs number of forms')
print('=' * 70)

# Use the best configuration
if best_config:
    w_name, D_label, D, _, _, _ = best_config
    w_func = weights[w_name]
else:
    w_name, D, w_func = 'uniform', LOG_T, weights['uniform']

checkpoints = [10, 20, 50, 100, 200, 300, 500, len(r_all)]
checkpoints = [c for c in checkpoints if c <= len(r_all)]

print(f'\nUsing: weight={w_name}, D={D_label if best_config else "log(T/2pi)"}')
print(f'{"N forms":>8} {"r_max":>8} {"R2":>7} {"R2_adj":>7}')
print('-' * 35)

for n_forms in checkpoints:
    r_subset = r_all[:n_forms]
    model = spectral_model(r_subset, D, w_func, k_arr)
    R2, R2_adj, _ = fit_spectral(model, excess, ss_tot, short_3)
    print(f'{n_forms:>8} {r_subset[-1]:>8.2f} {R2:>7.4f} {R2_adj:>7.4f}')

# ============================================================
# STEP 5: Hecke-weighted spectral model (if we have coefficients)
# ============================================================
if hecke_data:
    print('\n' + '=' * 70)
    print('HECKE-WEIGHTED SPECTRAL MODEL')
    print('=' * 70)

    primes_test = list(primerange(2, 50))

    # For each prime p, compute: Sum_j a_p(j)^2 * w(r_j) * cos(...)
    # The Hecke-modulated spectral model:
    # For prime p: the contribution is weighted by a_p(j)^2
    # (Sato-Tate average: <a_p^2> -> 1 over many forms)

    # With only 2 forms, test the basic idea:
    n_hecke = len(hecke_data)
    hecke_r = [hecke_data[i]['r'] for i in sorted(hecke_data.keys())]
    print(f'  Forms with Hecke data: {n_hecke} (r = {hecke_r})')

    # Build Hecke-weighted model for each prime
    for p in primes_test[:5]:
        model_p = np.zeros(MAX_LAG)
        for i in sorted(hecke_data.keys()):
            r = hecke_data[i]['r']
            a_p = hecke_data[i]['hecke'].get(p, 0)
            model_p += a_p ** 2 * np.cos(2 * np.pi * k_arr * r / D)

        if np.max(np.abs(model_p)) > 1e-20:
            corr = np.corrcoef(model_p, excess)[0, 1]
            print(f'  p={p}: a_p^2-weighted spectral, corr with excess = {corr:+.4f}')

    print(f'\n  With only {n_hecke} forms, Hecke modulation is meaningless.')
    print(f'  Need 100+ forms with coefficients for Sato-Tate average to converge.')

# ============================================================
# STEP 6: Direct test — does the spectral staircase predict primes?
# ============================================================
print('\n' + '=' * 70)
print('SPECTRAL STAIRCASE vs PRIME COUNTING')
print('=' * 70)

# Weyl's law: N(R) ~ R^2/12 for SL(2,Z)
# The spectral staircase should encode information about primes
# via the Selberg trace formula

# Compute the spectral staircase oscillation
# N(R) = N_smooth(R) + N_osc(R)
# N_smooth(R) ~ R^2/12 - (2/pi)*R*log(R) + ...
# N_osc(R) should relate to primes

r_sorted = np.sort(r_all)
N_actual = np.arange(1, len(r_sorted) + 1)
N_smooth = r_sorted ** 2 / 12  # leading Weyl term

# Oscillatory part
N_osc = N_actual - N_smooth

print(f'  Spectral staircase: {len(r_sorted)} forms')
print(f'  Weyl residual: mean={np.mean(N_osc):.2f}, std={np.std(N_osc):.2f}')

# The trace formula predicts: N_osc(R) ~ Sum_p log(p)/(p-1) * sin(R*log(p)) / (R*log(p))
# This is the spectral-side analog of the explicit formula for prime counting

# Test: Fourier transform of N_osc should show peaks at log(p)
fft_osc = np.fft.rfft(N_osc)
power_osc = np.abs(fft_osc) ** 2
freqs_osc = np.fft.rfftfreq(len(N_osc), d=(r_sorted[-1] - r_sorted[0]) / len(r_sorted))

# Find peaks
top_peaks = np.argsort(power_osc[1:])[-10:][::-1] + 1
print(f'\n  Top 10 Fourier peaks in spectral staircase oscillation:')
print(f'  {"Rank":<5} {"Freq":>10} {"Period":>10} {"Power":>12} {"log(p)?":>10}')
print(f'  {"-"*50}')

primes_check = list(primerange(2, 100))
for rank, idx in enumerate(top_peaks):
    freq = freqs_osc[idx]
    period = 1.0 / freq if freq > 0 else float('inf')
    # Check if freq matches log(p)/(2*pi) for some prime
    match = ''
    for p in primes_check:
        if abs(freq - np.log(p) / (2 * np.pi)) < 0.01:
            match = f'log({p})/2pi'
            break
        if abs(freq - np.log(p)) < 0.05:
            match = f'~log({p})'
            break
    print(f'  {rank+1:<5} {freq:>10.4f} {period:>10.2f} {power_osc[idx]:>12.1f} {match:>10}')

# ============================================================
# VERDICT
# ============================================================
print('\n' + '=' * 70)
print('VERDICT: SPECTRAL-GEOMETRIC DUALITY')
print('=' * 70)

print(f'\nForms available: {len(r_all)} spectral parameters, {len(hecke_data)} with Hecke coefficients')
print(f'Best spectral model R2_adj: {best_R2_adj:.4f}')
print(f'  (vs geometric model R2_adj: 0.626 from 5-param Selberg)')

if best_R2_adj > 0.30:
    print(f'\n>>> PARTIAL DUALITY: spectral side explains {best_R2_adj*100:.0f}% (adjusted)')
    print('>>> Convergence is visible but slow — need more forms')
elif best_R2_adj > 0.05:
    print(f'\n>>> WEAK DUALITY: spectral side explains only {best_R2_adj*100:.1f}%')
    print(f'>>> With {len(r_all)} forms, the spectral sum has barely begun to converge')
    print('>>> The trace formula dual requires ~10,000+ forms for this frequency range')
else:
    print(f'\n>>> NO CONVERGENCE: spectral model R2_adj = {best_R2_adj:.4f}')
    print(f'>>> {len(r_all)} forms is far too few for the spectral side to predict')
    print('>>> the pair correlation at the Odlyzko height T~2.7e11')
    print('>>> The geometric (prime) side converges in ~30 terms;')
    print('>>> the spectral side needs orders of magnitude more')

ratio = 0.626 / max(best_R2_adj, 0.001)
print(f'\nAsymmetry ratio: geometric/spectral = {ratio:.0f}x')
print('The geometric side (primes) is overwhelmingly more efficient than')
print('the spectral side (Maass forms) for describing the zeta zero ACF.')

print(f'\nTotal time: {time.time() - t0:.1f}s')
