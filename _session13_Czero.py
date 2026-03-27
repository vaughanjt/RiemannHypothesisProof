"""Quick check: V(g)=1 iff C(g/2)=0. Where is the first zero of C(tau)?"""
import numpy as np

for N in [50, 200, 500, 1000]:
    p = (1.0/np.arange(1,N+1)); p /= p.sum()
    w = np.log(np.arange(2,N+2))
    m2 = np.dot(p, w**2)
    g_bar = np.pi / np.sqrt(m2)

    # Fine search for first zero of C(tau)
    tau = np.linspace(0.01, 2*g_bar, 100000)
    C = np.dot(np.cos(np.outer(tau, w)), p)
    sign_changes = np.where(C[:-1]*C[1:] < 0)[0]

    if len(sign_changes) > 0:
        tau0 = tau[sign_changes[0]]
        g_star = 2*tau0
        print(f"N={N:>4}: tau0={tau0:.6f}, g*=2*tau0={g_star:.6f}, "
              f"g*/g_bar={g_star/g_bar:.4f}, g_bar={g_bar:.6f}")

    # Also check: what fraction of the Rayleigh CDF lies below g*?
    # For normalized gap g/g_bar ~ Rayleigh(scale), the CDF is 1 - exp(-x^2/(2s^2))
    # But gap distribution is NOT exactly Rayleigh. Just note the ratio.
    print(f"       g_bar*sqrt(m2)/pi = {g_bar*np.sqrt(m2)/np.pi:.6f} (should be 1)")

    # The "effective support" of the gap distribution
    # For a GP, P(g > c*g_bar) ~ exp(-const * c^2) for large c
    # At g* ~ 1.175*g_bar, this is P(g>1.175*g_bar)
    # For Rayleigh with mean = g_bar*sqrt(pi/2): scale = g_bar*sqrt(2/pi)
    # P(X > 1.175*g_bar) = exp(-(1.175*g_bar)^2/(2*s^2)) = exp(-1.175^2*pi/4)
    # = exp(-1.085) = 0.338
    from scipy.stats import rayleigh
    s = g_bar * np.sqrt(2/np.pi)  # Rayleigh scale with correct mean
    p_tail_rayleigh = 1 - rayleigh.cdf(g_star, scale=s)
    print(f"       Rayleigh P(g > g*) estimate = {p_tail_rayleigh:.4f}")
