"""Deep dive into the d=30 primorial structure and the 2D density-winding space.

The d=30 image shows 8 spiral arms = the 8 coprime residues mod 30.
Now: systematically map the FULL 2D parameter space (density, winding)
to find where hidden structures emerge, transitions happen, and
novel patterns appear that aren't visible at any single parameter setting.

ALSO: Analyze the 8-arm structure quantitatively:
- Which arms are densest? (connects to Chebyshev bias)
- Do arm densities match our operator coupling constants?
- What happens at the TRANSITIONS between arm counts?

AND: Color-code by prime residue class to make the mode structure visible.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sympy import primerange
import os, time

t0 = time.time()

# Setup
count = 50000
primes = list(primerange(2, count * 20))[:count]
primes_arr = np.array(primes, dtype=float)
n_arr = np.arange(1, count + 1, dtype=float)

os.makedirs('ulam_explore', exist_ok=True)


def save_spiral(density, winding=20000, n_points=None, filename=None, title=None):
    if n_points is None: n_points = count
    x, y = gen_spiral(density, winding, n_points)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    ax.scatter(x, y, s=0.01, c='blue', alpha=0.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)
    if filename is None: filename = f'ulam_d{density:.2f}.png'
    plt.savefig(filename, bbox_inches='tight', facecolor='white')
    plt.close()
    return filename


def gen_spiral(density, winding=20000, n_pts=None):
    if n_pts is None: n_pts = count
    n = n_arr[:n_pts]; p = primes_arr[:n_pts]
    x = (n * np.sin(p / density)) / winding
    y = (n * np.cos(p / density)) / winding
    return x, y


# ============================================================
# 1. Color-coded d=30 spiral by residue class
# ============================================================
print("1. Color-coded residue spirals at d=30...", flush=True)

residues_30 = [1, 7, 11, 13, 17, 19, 23, 29]
colors_8 = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8',
            '#f58231', '#911eb4', '#42d4f4', '#f032e6']

fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)

for ax_idx, (density, title) in enumerate([(30, 'd=30 (primorial 2*3*5)'),
                                            (6, 'd=6 (primorial 2*3)'),
                                            (210, 'd=210 (primorial 2*3*5*7)')]):
    ax = axes[ax_idx]
    n_pts = 45000
    mod_val = int(density) if density in [6, 30, 210] else 30

    for i, r in enumerate(sorted([r for r in range(1, mod_val) if np.gcd(r, mod_val) == 1])):
        mask = np.array([p % mod_val == r for p in primes[:n_pts]])
        n_sub = n_arr[:n_pts][mask]
        p_sub = primes_arr[:n_pts][mask]
        x = (n_sub * np.sin(p_sub / density)) / 20000
        y = (n_sub * np.cos(p_sub / density)) / 20000
        c = colors_8[i % len(colors_8)]
        ax.scatter(x, y, s=0.02, c=c, alpha=0.6, label=f'{r} mod {mod_val}')

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

# Legend for first plot
axes[0].legend(loc='upper right', fontsize=5, markerscale=20, ncol=2)

plt.tight_layout()
plt.savefig('ulam_explore/residue_colored_primorials.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: residue_colored_primorials.png", flush=True)


# ============================================================
# 2. Arm density analysis at d=30
# ============================================================
print("\n2. Arm density analysis at d=30...", flush=True)

# Count primes in each residue class and measure their angular spread
density = 30
n_pts = 45000

arm_stats = {}
for r in residues_30:
    mask = np.array([p % 30 == r for p in primes[:n_pts]])
    p_sub = primes_arr[:n_pts][mask]
    # Angular position
    angles = (p_sub / density) % (2 * np.pi)
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    angular_spread = np.std(np.unwrap(angles))
    arm_stats[r] = {
        'count': mask.sum(),
        'fraction': mask.sum() / n_pts,
        'mean_angle': mean_angle,
        'angular_spread': angular_spread,
    }

print(f"\n  {'Residue':>8} {'Count':>8} {'Fraction':>10} {'Mean angle':>12} {'Spread':>10}")
print(f"  {'-'*52}")
for r in residues_30:
    s = arm_stats[r]
    print(f"  {r:>8} {s['count']:>8} {s['fraction']:>10.4f} "
          f"{s['mean_angle']:>+12.4f} {s['angular_spread']:>10.4f}")

# Chebyshev bias: are 3 mod 4 primes more common than 1 mod 4?
mod4_counts = {1: sum(1 for p in primes[:n_pts] if p%4==1),
               3: sum(1 for p in primes[:n_pts] if p%4==3)}
print(f"\n  Chebyshev bias (mod 4): 1mod4={mod4_counts[1]}, 3mod4={mod4_counts[3]}, "
      f"ratio={mod4_counts[3]/mod4_counts[1]:.4f}")


# ============================================================
# 3. The 2D parameter space: density vs winding
# ============================================================
print("\n3. Mapping 2D parameter space (density x winding)...", flush=True)

def angular_entropy(density, winding=20000, n_pts=20000):
    x, y = gen_spiral(density, winding, n_pts)
    theta = np.arctan2(y, x)
    hist, _ = np.histogram(theta, bins=180, range=(-np.pi, np.pi))
    hist = hist / hist.sum()
    ent = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))
    return ent / np.log(180)

# Scan density x count_used (proxy for winding/scale interaction)
densities = np.concatenate([np.arange(1, 50, 1), np.arange(50, 250, 5),
                            np.arange(250, 510, 10)])
n_pts_vals = [5000, 10000, 20000, 40000]

entropy_map = np.zeros((len(densities), len(n_pts_vals)))

for i, d in enumerate(densities):
    for j, np_val in enumerate(n_pts_vals):
        entropy_map[i, j] = angular_entropy(d, n_pts=np_val)

fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
im = ax.imshow(entropy_map.T, aspect='auto', origin='lower',
               extent=[densities[0], densities[-1], 0, len(n_pts_vals)-1],
               cmap='viridis_r')  # reversed: dark = structured
ax.set_xlabel('Density')
ax.set_ylabel('Scale (n_points)')
ax.set_yticks(range(len(n_pts_vals)))
ax.set_yticklabels([str(n) for n in n_pts_vals])
ax.set_title('Angular Entropy Map (dark = more structured)')
plt.colorbar(im, label='Normalized entropy')
plt.savefig('ulam_explore/entropy_2d_map.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: entropy_2d_map.png", flush=True)

# Find the deepest entropy valleys
min_indices = np.unravel_index(np.argsort(entropy_map.ravel())[:15], entropy_map.shape)
print(f"\n  Top 15 most structured (density, n_pts):")
for k in range(15):
    d_idx, n_idx = min_indices[0][k], min_indices[1][k]
    print(f"    d={densities[d_idx]:>6.1f}, n={n_pts_vals[n_idx]:>6}, "
          f"entropy={entropy_map[d_idx, n_idx]:.4f}")


# ============================================================
# 4. Phase transitions: where do arms split/merge?
# ============================================================
print("\n4. Phase transitions in arm count vs density...", flush=True)

def count_arms(density, n_pts=30000):
    x, y = gen_spiral(density, n_pts=n_pts)
    theta = np.arctan2(y, x)
    hist, _ = np.histogram(theta, bins=360, range=(-np.pi, np.pi))
    smoothed = np.convolve(hist, np.ones(5)/5, mode='same')
    threshold = np.mean(smoothed) + 1.5 * np.std(smoothed)
    peaks = np.sum((smoothed[1:-1] > smoothed[:-2]) &
                    (smoothed[1:-1] > smoothed[2:]) &
                    (smoothed[1:-1] > threshold))
    return peaks

# Fine sweep
d_fine = np.arange(0.5, 100, 0.25)
arm_counts = [count_arms(d) for d in d_fine]

fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
ax.plot(d_fine, arm_counts, 'b-', linewidth=0.5)
ax.set_xlabel('Density')
ax.set_ylabel('Number of arms')
ax.set_title('Arm count vs density (phase transitions)')
# Mark primorials
for val, label in [(2, '2'), (6, '2*3'), (30, '2*3*5')]:
    ax.axvline(val, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(val, max(arm_counts)*0.9, label, fontsize=7, color='red')
plt.savefig('ulam_explore/arm_count_vs_density.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: arm_count_vs_density.png", flush=True)


# ============================================================
# 5. Novel structures: density near golden ratio, sqrt(2), etc.
# ============================================================
print("\n5. Irrational density values...", flush=True)

phi = (1 + np.sqrt(5)) / 2  # golden ratio
specials = [
    (phi, "golden ratio"),
    (np.sqrt(2), "sqrt(2)"),
    (np.sqrt(3), "sqrt(3)"),
    (np.pi / phi, "pi/phi"),
    (np.e / np.pi, "e/pi"),
    (np.log(2), "ln(2)"),
    (1 / np.log(2), "1/ln(2)"),
    (np.pi * np.e, "pi*e"),
    (2 * np.pi / np.log(2), "2pi/ln2"),
]

for val, name in specials:
    m = angular_entropy(val, n_pts=30000)
    n_arms = count_arms(val)
    save_spiral(val, n_points=45000,
                filename=f'ulam_explore/special_{name.replace("/","_").replace("*","x")}.png',
                title=f'{name} = {val:.6f}')
    print(f"  {name:>12} ({val:.6f}): entropy={m:.4f}, arms={n_arms}", flush=True)


# ============================================================
# 6. Multi-panel comparison: the primorial sequence
# ============================================================
print("\n6. Primorial sequence comparison...", flush=True)

primorials = [(2, '2'), (6, '2*3'), (30, '2*3*5'), (210, '2*3*5*7')]
fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)

for idx, (d, label) in enumerate(primorials):
    ax = axes[idx]
    x, y = gen_spiral(d, n_pts=45000)
    ax.scatter(x, y, s=0.01, c='blue', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title(f'd={label} ({d})', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig('ulam_explore/primorial_sequence.png', bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: primorial_sequence.png", flush=True)


# ============================================================
# 7. Connection to operator: arm density vs coupling constants
# ============================================================
print("\n7. Arm density vs operator coupling...", flush=True)

# Our operator found: mod 8 couplings [1.22, 3.47, 0.001, 1.61] for [1,3,5,7]
operator_couplings = {1: 1.22, 3: 3.47, 5: 0.001, 7: 1.61}

# Measure arm density in the d=8 spiral
print(f"\n  {'r mod 8':>8} {'Arm density':>12} {'Op coupling':>12} {'Product':>10}")
print(f"  {'-'*46}")

for r in [1, 3, 5, 7]:
    mask = np.array([p % 8 == r for p in primes[:30000]])
    p_sub = primes_arr[:30000][mask]
    # Angular concentration at d=8
    angles = (p_sub / 8) % (2 * np.pi)
    hist, _ = np.histogram(angles, bins=90, range=(0, 2*np.pi))
    max_concentration = hist.max() / hist.mean()
    print(f"  {r:>8} {max_concentration:>12.4f} {operator_couplings[r]:>12.4f} "
          f"{max_concentration * operator_couplings[r]:>10.4f}")


# ============================================================
# 8. Density = log(p_k) / (2*pi) — the natural scale from the operator
# ============================================================
print("\n8. Operator-natural density values...", flush=True)

# In our operator, the prime frequencies are f_p = log(p) * mean_density
# The spiral angle is Prime(n) / density
# For the spiral to resonate with prime p, we need:
#   p / density ~ 2*pi*k for some integer k
# i.e., density ~ p / (2*pi*k)
# The natural density for prime p is p/(2*pi)

for p_val in [2, 3, 5, 7, 11, 13, 29, 41]:
    d_natural = p_val / (2 * np.pi)
    m = angular_entropy(d_natural, n_pts=30000)
    n_arms = count_arms(d_natural)
    save_spiral(d_natural, n_points=45000,
                filename=f'ulam_explore/natural_p{p_val}.png',
                title=f'd=p/{2:.0f}pi, p={p_val} (d={d_natural:.4f})')
    print(f"  p={p_val:>3}, d={d_natural:.4f}: entropy={m:.4f}, arms={n_arms}", flush=True)


def save_spiral(density, winding=20000, n_points=None, filename=None, title=None):
    if n_points is None: n_points = count
    x, y = gen_spiral(density, winding, n_points)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    ax.scatter(x, y, s=0.01, c='blue', alpha=0.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=10)
    if filename is None: filename = f'ulam_d{density:.2f}.png'
    plt.savefig(filename, bbox_inches='tight', facecolor='white')
    plt.close()
    return filename


print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)
