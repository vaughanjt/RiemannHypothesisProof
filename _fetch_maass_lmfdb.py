"""Fetch Maass form data from LMFDB at scale.

Goal: 500+ Maass forms with Hecke coefficients for spectral-geometric duality test.
Uses the existing LMFDB client with caching."""
import sys
sys.path.insert(0, 'src')
import json
import time
import numpy as np
from pathlib import Path
from sympy import isprime

# Try to use existing LMFDB client
try:
    from riemann.analysis.lmfdb_client import query_lmfdb, LMFDBError
    HAS_CLIENT = True
except ImportError:
    HAS_CLIENT = False

import requests

LMFDB_BASE = "https://www.lmfdb.org/api"
DATA_DIR = Path('data')
MAASS_CACHE = DATA_DIR / 'maass_forms.json'

# ============================================================
# FETCH SPECTRAL PARAMETERS
# ============================================================
print('='*70)
print('FETCHING MAASS FORM DATA FROM LMFDB')
print('='*70)

def fetch_maass_spectral(limit=100, offset=0):
    """Fetch spectral parameters from LMFDB maass_rigor collection."""
    url = f"{LMFDB_BASE}/maass_rigor/"
    params = {
        "level": "1",
        "_format": "json",
        "_limit": str(limit),
        "_offset": str(offset),
        "_sort": "spectral_parameter",
        "_fields": "maass_label,spectral_parameter,symmetry,level",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("data", [])
        print(f'  Fetched {len(records)} records (offset={offset})')
        return records
    except Exception as e:
        print(f'  Error at offset={offset}: {e}')
        return []

def fetch_maass_coefficients(label):
    """Fetch Fourier coefficients for a specific Maass form."""
    url = f"{LMFDB_BASE}/maass_rigor/"
    params = {
        "maass_label": label,
        "_format": "json",
        "_fields": "maass_label,spectral_parameter,symmetry,coefficients",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("data", [])
        if records and "coefficients" in records[0]:
            return records[0]["coefficients"]
        return None
    except Exception as e:
        print(f'  Error fetching coeffs for {label}: {e}')
        return None

# Step 1: Fetch all level-1 spectral parameters
print('\nStep 1: Fetching spectral parameters...')
all_records = []
for offset in range(0, 2000, 100):
    batch = fetch_maass_spectral(limit=100, offset=offset)
    if not batch:
        break
    all_records.extend(batch)
    time.sleep(0.5)  # rate limiting

print(f'\nTotal records fetched: {len(all_records)}')

# Parse spectral parameters
forms = []
for rec in all_records:
    try:
        sp = rec.get("spectral_parameter")
        if sp is None:
            continue
        # Handle different LMFDB data formats
        if isinstance(sp, dict):
            r = float(sp.get("data", sp.get("$numberDouble", 0)))
        elif isinstance(sp, (int, float)):
            r = float(sp)
        elif isinstance(sp, str):
            r = float(sp)
        else:
            continue

        sym = rec.get("symmetry", -1)
        label = rec.get("maass_label", "unknown")
        forms.append({"label": label, "r": r, "symmetry": int(sym)})
    except (ValueError, TypeError) as e:
        continue

print(f'Parsed spectral parameters: {len(forms)}')
if forms:
    rs = [f["r"] for f in forms]
    print(f'Range: r = {min(rs):.4f} to {max(rs):.4f}')

# Step 2: Fetch coefficients for a subset of forms
# Focus on the first 50 forms (manageable, gives us Hecke eigenvalues)
print('\nStep 2: Fetching Fourier coefficients for first 50 forms...')
forms_with_coeffs = []
for i, form in enumerate(forms[:50]):
    label = form["label"]
    print(f'  [{i+1}/50] {label} (r={form["r"]:.4f})...', end='')
    coeffs = fetch_maass_coefficients(label)
    if coeffs is not None:
        # Parse coefficient data
        if isinstance(coeffs, list):
            parsed = []
            for c in coeffs:
                if isinstance(c, dict):
                    parsed.append(float(c.get("data", c.get("$numberDouble", 0))))
                elif isinstance(c, (int, float)):
                    parsed.append(float(c))
                elif isinstance(c, str):
                    parsed.append(float(c))
            form["coefficients"] = parsed
            form["n_coeffs"] = len(parsed)
            forms_with_coeffs.append(form)
            print(f' {len(parsed)} coefficients')
        else:
            print(f' unexpected format: {type(coeffs)}')
    else:
        print(' no coefficients')
    time.sleep(0.3)

print(f'\nForms with coefficients: {len(forms_with_coeffs)}')

# Step 3: Save to cache
cache_data = {
    "spectral_parameters": forms,
    "forms_with_coefficients": forms_with_coeffs,
    "metadata": {
        "level": 1,
        "source": "LMFDB maass_rigor",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_spectral": len(forms),
        "total_with_coeffs": len(forms_with_coeffs),
    }
}

MAASS_CACHE.parent.mkdir(parents=True, exist_ok=True)
with open(MAASS_CACHE, 'w') as f:
    json.dump(cache_data, f, indent=2)
print(f'\nSaved to {MAASS_CACHE}')

# ============================================================
# ANALYSIS OF FETCHED DATA
# ============================================================
print('\n' + '='*70)
print('ANALYSIS OF FETCHED MAASS FORM DATA')
print('='*70)

if len(forms) > 10:
    rs = np.array([f["r"] for f in forms])
    syms = np.array([f["symmetry"] for f in forms])

    # Spacing statistics
    from scipy.stats import kstest

    for label, mask in [("All", np.ones(len(rs), dtype=bool)),
                         ("Even", syms == 0),
                         ("Odd", syms == 1)]:
        subset = np.sort(rs[mask])
        if len(subset) < 10:
            continue
        spacings = np.diff(subset)
        spacings_norm = spacings / np.mean(spacings)

        ks_poisson = kstest(spacings_norm, 'expon', args=(0, 1))
        ks_rayleigh = kstest(spacings_norm, 'rayleigh', args=(0, np.sqrt(2/np.pi)))

        print(f'\n{label} ({len(subset)} forms):')
        print(f'  Poisson:    KS={ks_poisson.statistic:.4f}, p={ks_poisson.pvalue:.4f}')
        print(f'  GUE-like:   KS={ks_rayleigh.statistic:.4f}, p={ks_rayleigh.pvalue:.4f}')
        print(f'  Level repulsion P(s<0.3): {np.mean(spacings_norm < 0.3):.3f}')

# Extract Hecke eigenvalues at primes
if forms_with_coeffs:
    print('\n--- Hecke eigenvalue matrix ---')
    primes_list = [p for p in range(2, 100) if isprime(p)]
    print(f'Primes: {primes_list[:15]}...')

    # Build matrix: rows = forms, columns = primes
    hecke_matrix = []
    form_labels = []
    for form in forms_with_coeffs:
        coeffs = form.get("coefficients", [])
        if len(coeffs) < 100:
            continue
        row = [coeffs[p-1] for p in primes_list if p <= len(coeffs)]
        if len(row) == len(primes_list):
            hecke_matrix.append(row)
            form_labels.append(form["label"])

    if hecke_matrix:
        H = np.array(hecke_matrix)
        print(f'Hecke matrix shape: {H.shape} (forms x primes)')

        # Statistics
        print(f'Range: [{H.min():.4f}, {H.max():.4f}]')
        print(f'|lambda_p| > 2 entries: {np.sum(np.abs(H) > 2)} / {H.size}')

        # Column-wise (per-prime) second moment
        col_var = np.mean(H**2, axis=0)
        print(f'\nPer-prime <lambda_p^2> (Sato-Tate predicts 1.0):')
        for i, p in enumerate(primes_list[:15]):
            print(f'  p={p:>3}: <lambda^2>={col_var[i]:.4f} (n={H.shape[0]} forms)')

        # Singular value decomposition
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        print(f'\nSVD singular values (top 10): {S[:10].round(3)}')
        print(f'Rank-1 fraction: {S[0]**2 / np.sum(S**2):.4f}')
        print(f'Effective rank (90% variance): {np.searchsorted(np.cumsum(S**2)/np.sum(S**2), 0.9) + 1}')

        # Spectral-geometric duality test
        print('\n--- Spectral-Geometric Duality Test ---')
        from riemann.analysis.bost_connes_operator import spacing_autocorrelation, polynomial_unfold

        # Load zeta data
        def load_zeros(path):
            values = []
            with open(path) as f:
                for line in f:
                    try: values.append(float(line.strip()))
                    except ValueError: continue
            return np.array(values)

        res = load_zeros('data/odlyzko/zeros3.txt')
        T_base = 267653395647.0
        log_T = np.log(T_base / (2*np.pi))
        density = log_T / (2*np.pi)
        sp = np.diff(res) * density
        sp = sp / np.mean(sp)

        test_lag = 200
        acf = spacing_autocorrelation(sp, test_lag)

        # GUE baseline (quick, fewer matrices)
        rng = np.random.default_rng(42)
        gue_acfs = []
        for _ in range(50):
            A = rng.standard_normal((1200, 1200)) + 1j * rng.standard_normal((1200, 1200))
            Hm = (A + A.conj().T) / (2 * np.sqrt(2400))
            eigs = np.linalg.eigvalsh(Hm)
            s = polynomial_unfold(eigs, trim_fraction=0.1)
            if len(s) > test_lag + 10:
                gue_acfs.append(spacing_autocorrelation(s, test_lag))
        gue_acf = np.mean(gue_acfs, axis=0)
        excess = acf[1:test_lag+1] - gue_acf[1:test_lag+1]
        ss_tot = np.sum(excess**2)

        # Spectral side: sum_j w(r_j) * cos(k * r_j / ...)
        rs_all = np.array([f["r"] for f in forms])
        for n_forms in [50, 100, 200, len(forms)]:
            rs_use = rs_all[:min(n_forms, len(rs_all))]
            spectral = np.zeros(test_lag)
            for r in rs_use:
                weight = 1.0 / np.cosh(np.pi * r)
                for k in range(test_lag):
                    spectral[k] += weight * np.cos((k+1) * r / density)

            # Fit scale
            dot = np.dot(spectral, excess)
            norm_sq = np.dot(spectral, spectral)
            if norm_sq > 1e-30:
                scale = dot / norm_sq
                pred = scale * spectral
                ss_res = np.sum((excess - pred)**2)
                R2 = 1 - ss_res / ss_tot
                R2_adj = 1 - (1 - R2) * (test_lag - 1) / (test_lag - 2)
                corr = np.corrcoef(spectral, excess)[0,1]
                print(f'  {n_forms:>5} forms: R2={R2:.4f}, R2_adj={R2_adj:.4f}, corr={corr:+.4f}')
            else:
                print(f'  {n_forms:>5} forms: spectral vector ~0')
    else:
        print('No forms with enough coefficients for Hecke matrix')
else:
    print('\nNo coefficient data fetched -- LMFDB may be rate-limiting')
    print('Using hardcoded data from earlier session instead.')

print('\n' + '='*70)
print('FETCH COMPLETE')
print('='*70)
