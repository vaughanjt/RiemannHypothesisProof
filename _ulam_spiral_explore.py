"""Ulam spiral exploration: adjustable winding reveals hidden prime structure.

The parametric spiral:
  point(n) = (n/winding) * (sin(Prime(n)/density), cos(Prime(n)/density))

The angle is Prime(n)/density, NOT n. So:
- When density ~ p_k (a specific prime), primes near multiples of density
  create angular clustering -> radial lines appear
- When density = 2*pi*k for integer k, the spiral has k-fold symmetry
- When density is irrational * pi, the pattern is quasi-random

By sweeping density, we can discover which values create the sharpest
line structures, revealing the arithmetic progressions that generate them.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sympy import primerange
import time

t0 = time.time()

# Generate primes
print("Generating primes...", flush=True)
count = 50000
primes = list(primerange(2, count * 20))  # Need enough primes
primes = primes[:count]
primes_arr = np.array(primes, dtype=float)
n_arr = np.arange(1, count + 1, dtype=float)
print(f"  {count} primes up to {primes[-1]}", flush=True)


def generate_spiral(density, winding=20000, n_points=None):
    """Generate spiral points for given parameters."""
    if n_points is None:
        n_points = count
    n = n_arr[:n_points]
    p = primes_arr[:n_points]
    x = (n * np.sin(p / density)) / winding
    y = (n * np.cos(p / density)) / winding
    return x, y


def save_spiral(density, winding=20000, filename=None, title=None, n_points=None):
    """Save a spiral plot."""
    if n_points is None:
        n_points = count
    x, y = generate_spiral(density, winding, n_points)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    ax.scatter(x, y, s=0.01, c='blue', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_frame_on(True)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    else:
        ax.set_title(f'density={density:.4f}, winding={winding}', fontsize=10)

    if filename is None:
        filename = f'ulam_d{density:.2f}.png'
    plt.savefig(filename, bbox_inches='tight', facecolor='white')
    plt.close()
    return filename


def measure_structure(density, winding=20000, n_points=None):
    """Quantify the line structure in a spiral.

    Returns metrics:
    - angular_entropy: low = strong radial lines, high = uniform
    - radial_variance: how much the radial distribution varies by angle
    - n_rays: estimated number of visible radial lines
    """
    if n_points is None:
        n_points = count
    x, y = generate_spiral(density, winding, n_points)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)  # [-pi, pi]

    # Angular histogram
    n_bins = 360
    hist, edges = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
    hist = hist / hist.sum()

    # Entropy (lower = more structured)
    entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))
    max_entropy = np.log(n_bins)
    norm_entropy = entropy / max_entropy  # 0 = one ray, 1 = uniform

    # Angular FFT to detect n-fold symmetry
    fft_angular = np.abs(np.fft.rfft(hist))
    # Skip DC, find dominant frequency
    fft_angular[0] = 0
    dominant_freq = np.argmax(fft_angular[1:]) + 1
    symmetry_strength = fft_angular[dominant_freq] / np.mean(fft_angular[1:])

    # Estimate number of rays: count peaks in angular histogram
    smoothed = np.convolve(hist, np.ones(5)/5, mode='same')
    threshold = np.mean(smoothed) + 2 * np.std(smoothed)
    peaks = np.sum((smoothed[1:-1] > smoothed[:-2]) &
                    (smoothed[1:-1] > smoothed[2:]) &
                    (smoothed[1:-1] > threshold))

    return {
        'entropy': norm_entropy,
        'dominant_symmetry': dominant_freq,
        'symmetry_strength': symmetry_strength,
        'n_rays': peaks,
    }


# ============================================================
# SWEEP 1: Density from 1 to 500
# ============================================================
print("\nSWEEP 1: Density from 1 to 500", flush=True)
print(f"  {'density':>10} {'entropy':>8} {'symmetry':>10} {'strength':>10} {'n_rays':>8}", flush=True)
print(f"  {'-'*50}", flush=True)

interesting = []

for d in np.concatenate([
    np.arange(1, 20, 0.5),      # Fine sweep 1-20
    np.arange(20, 100, 2),       # Medium sweep 20-100
    np.arange(100, 500, 10),     # Coarse sweep 100-500
    [np.pi, 2*np.pi, 3*np.pi, np.e, np.pi**2, np.sqrt(2)*10,
     6, 10, 15, 30, 41, 42, 50, 100, 150, 200],  # Special values
]):
    metrics = measure_structure(d, n_points=30000)
    is_interesting = metrics['entropy'] < 0.92 or metrics['symmetry_strength'] > 5

    if is_interesting:
        interesting.append((d, metrics))
        print(f"  {d:>10.4f} {metrics['entropy']:>8.4f} {metrics['dominant_symmetry']:>10} "
              f"{metrics['symmetry_strength']:>10.2f} {metrics['n_rays']:>8} <<<", flush=True)

print(f"\n  Found {len(interesting)} interesting density values", flush=True)


# ============================================================
# Save images for the most interesting densities
# ============================================================
print("\nSaving images for top structures...", flush=True)

# Sort by entropy (most structured first)
interesting.sort(key=lambda x: x[1]['entropy'])

saved_files = []
for i, (d, m) in enumerate(interesting[:20]):
    fname = save_spiral(d, winding=20000, n_points=45000,
                        filename=f'ulam_explore/spiral_d{d:.4f}.png',
                        title=f'd={d:.4f}, sym={m["dominant_symmetry"]}, '
                              f'rays={m["n_rays"]}, ent={m["entropy"]:.3f}')
    saved_files.append((d, fname, m))


# ============================================================
# SWEEP 2: Fine sweep around pi and small integers
# ============================================================
print("\nSWEEP 2: Fine sweep around key values", flush=True)

key_values = {
    'pi': np.pi,
    '2*pi': 2*np.pi,
    'e': np.e,
    '6': 6.0,
    '10': 10.0,
    '30': 30.0,
    '41': 41.0,
    '150': 150.0,
}

for name, center in key_values.items():
    best_ent = 1.0
    best_d = center
    for delta in np.linspace(-0.5, 0.5, 51):
        d = center + delta
        if d <= 0:
            continue
        m = measure_structure(d, n_points=20000)
        if m['entropy'] < best_ent:
            best_ent = m['entropy']
            best_d = d

    m_best = measure_structure(best_d, n_points=30000)
    print(f"  Near {name:>6} ({center:.4f}): best d={best_d:.4f}, "
          f"entropy={m_best['entropy']:.4f}, "
          f"sym={m_best['dominant_symmetry']}, rays={m_best['n_rays']}", flush=True)


# ============================================================
# SWEEP 3: Multiples of pi (angular resonance)
# ============================================================
print("\nSWEEP 3: Multiples and fractions of pi", flush=True)

for k_num, k_den in [(1,1), (1,2), (1,3), (1,4), (1,6),
                      (2,1), (3,1), (4,1), (5,1), (6,1),
                      (3,2), (5,2), (7,2), (5,3), (7,3)]:
    d = k_num * np.pi / k_den
    m = measure_structure(d, n_points=30000)
    label = f"{k_num}*pi/{k_den}" if k_den > 1 else f"{k_num}*pi"
    print(f"  {label:>10} (d={d:>8.4f}): entropy={m['entropy']:.4f}, "
          f"sym={m['dominant_symmetry']}, rays={m['n_rays']}", flush=True)


# ============================================================
# SWEEP 4: Density = prime values
# ============================================================
print("\nSWEEP 4: Density = prime numbers", flush=True)

small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

for p in small_primes:
    m = measure_structure(float(p), n_points=30000)
    print(f"  p={p:>3}: entropy={m['entropy']:.4f}, "
          f"sym={m['dominant_symmetry']}, "
          f"strength={m['symmetry_strength']:.2f}, "
          f"rays={m['n_rays']}", flush=True)


# ============================================================
# Generate final gallery of the most striking patterns
# ============================================================
print("\nGenerating gallery of striking patterns...", flush=True)

import os
os.makedirs('ulam_explore', exist_ok=True)

gallery = [
    (2.0, "d=2 (smallest prime)"),
    (3.0, "d=3"),
    (np.pi, "d=pi"),
    (2*np.pi, "d=2*pi"),
    (5.0, "d=5"),
    (6.0, "d=6 (2*3)"),
    (7.0, "d=7"),
    (10.0, "d=10"),
    (30.0, "d=30 (2*3*5)"),
    (41.0, "d=41 (Euler prime)"),
    (150.0, "d=150 (original)"),
    (0.5, "d=0.5 (very wide)"),
    (1.0, "d=1 (wide)"),
    (500.0, "d=500 (very tight)"),
]

# Also add the top 6 most structured from the sweep
for d, m in interesting[:6]:
    if not any(abs(d - g[0]) < 0.1 for g in gallery):
        gallery.append((d, f"d={d:.2f} (sweep find)"))

for d, label in gallery:
    fname = save_spiral(d, winding=20000, n_points=45000,
                        filename=f'ulam_explore/gallery_{label.replace(" ","_").replace("*","x").replace("(","").replace(")","").replace("=","")}.png',
                        title=label)
    print(f"  Saved: {fname}", flush=True)


# ============================================================
# Summary
# ============================================================
print("\n" + "="*70, flush=True)
print("SUMMARY", flush=True)
print("="*70, flush=True)

print(f"\n  Total density values tested: ~{len(interesting) + 200}", flush=True)
print(f"  Interesting (low entropy): {len(interesting)}", flush=True)

print(f"\n  Top 10 most structured:", flush=True)
print(f"  {'density':>10} {'entropy':>8} {'symmetry':>10} {'rays':>6}", flush=True)
for d, m in interesting[:10]:
    print(f"  {d:>10.4f} {m['entropy']:>8.4f} {m['dominant_symmetry']:>10} {m['n_rays']:>6}", flush=True)

print(f"\n  Images saved in ulam_explore/", flush=True)
print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
