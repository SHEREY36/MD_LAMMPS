"""
Overlay Garzó–Dufty (GD) kinetic theory predictions on DEM data
for frictionless (μ=0) simple shear flow.

Theory: Chialvo & Sundaresan (2013), Eqs. (1)–(18), with
Torquato's g0 (Eq. 12) and φc = 0.636 for μ=0.

The Chialvo-scaled steady-state quantities are:
  p* = p / [ρs (γ̇d)²]  = J·H / K          (Eq. 16)
  η  = τ / p            = √(J·K) / H        (Eq. 18)
  T* = T / (γ̇d)²       = J / K              (Eq. 15)

These depend ONLY on φ and e (not on the dimensional shear rate),
so the comparison is apples-to-apples regardless of gdot.
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# GD Kinetic Theory (Chialvo 2013, Eqs. 4–12, 15–18)
# =============================================================

def g0_carnahan_starling(phi):
    """Carnahan-Starling RDF, Eq. (11)"""
    return (1.0 - phi / 2.0) / (1.0 - phi)**3

def g0_torquato(phi, phi_c=0.636, phi_f=0.49):
    """Torquato RDF, Eq. (12): CS below phi_f, divergent above"""
    phi = np.atleast_1d(phi).astype(float)
    g = np.empty_like(phi)
    lo = phi <= phi_f
    hi = ~lo
    g[lo] = g0_carnahan_starling(phi[lo])
    g_f = g0_carnahan_starling(phi_f)
    g[hi] = g_f * (phi_c - phi_f) / (phi_c - phi[hi])
    return g

def H_func(phi, e, g0):
    """Eq. (4): H = φ[1 + 2(1+e)φ g0]"""
    return phi * (1.0 + 2.0 * (1.0 + e) * phi * g0)

def eta_star_k(phi, e, g0):
    """Eq. (8): kinetic viscosity contribution"""
    num = 1.0 - (2.0/5.0) * (1.0 + e) * (1.0 - 3.0*e) * phi * g0
    den = (1.0 - 0.25*(1.0 - e)**2 - (5.0/24.0)*(1.0 - e**2)) * g0
    return num / den

def eta_star_c(phi, e, g0, ek):
    """Eq. (9): collisional viscosity contribution"""
    return (4.0/5.0) * (1.0 + e) * phi * g0 * ek

def eta_star_b(phi, e, g0):
    """Eq. (10): bulk viscosity contribution"""
    return (384.0 / (25.0 * np.pi)) * (1.0 + e) * phi**2 * g0

def J_func(phi, e, g0):
    """Eq. (5): J = (5√π / 96) η*"""
    ek = eta_star_k(phi, e, g0)
    ec = eta_star_c(phi, e, g0, ek)
    eb = eta_star_b(phi, e, g0)
    eta_total = ek + ec + eb
    return (5.0 * np.sqrt(np.pi) / 96.0) * eta_total

def K_func(phi, e, g0):
    """Eq. (6): K = (12/√π) φ² g0 (1 - e²)"""
    return (12.0 / np.sqrt(np.pi)) * phi**2 * g0 * (1.0 - e**2)

def gd_steady_state(phi, e, phi_c=0.636):
    """
    Compute GD steady-state Chialvo-scaled quantities.
    
    Returns dict with:
      pScaled  = p / [ρs (γ̇d)²]  = J·H / K
      eta      = τ / p            = √(J·K) / H
      TScaled  = T / (γ̇d)²       = J / K
    """
    phi = np.atleast_1d(phi).astype(float)
    
    g0 = g0_torquato(phi, phi_c=phi_c)
    H = H_func(phi, e, g0)
    J = J_func(phi, e, g0)
    K = K_func(phi, e, g0)
    
    # Avoid division by zero at φ=0 where K=0
    with np.errstate(divide='ignore', invalid='ignore'):
        TScaled = np.where(K > 0, J / K, np.inf)
        pScaled = np.where(K > 0, J * H / K, np.inf)
        eta = np.where(H > 0, np.sqrt(J * K) / H, 0.0)
    
    return {
        'pScaled': pScaled,
        'tau_over_p': eta,
        'TScaled': TScaled,
        'g0': g0, 'H': H, 'J': J, 'K': K,
    }

# =============================================================
# CLI
# =============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Plot Chialvo-style DEM data vs GD theory.")
parser.add_argument("--base-dir", default=os.path.join(SCRIPT_DIR, "jobsubmit5"),
                    help="Directory containing e_xxx/phi_xx folders")
parser.add_argument("--out-dir", default=None,
                    help="Directory to save plots (default: base-dir)")
args = parser.parse_args()

# =============================================================
# DEM Data Extraction
# =============================================================

base_dir = args.base_dir
out_dir = args.out_dir if args.out_dir is not None else base_dir
os.makedirs(out_dir, exist_ok=True)

e_cases  = ["e_099", "e_095", "e_090", "e_080", "e_070"]
phi_cases = ["phi_01", "phi_05", "phi_10", "phi_20", "phi_30",
             "phi_40", "phi_49", "phi_50", "phi_55", "phi_60"]

steady_fraction = 0.3

# Flip detection
flip_threshold = 0.0
flip_window    = 0

def get_flip_mask(tilt_xy, Lx_estimate=None):
    n = len(tilt_xy)
    if n < 2:
        return np.ones(n, dtype=bool)
    dtilt = np.diff(tilt_xy)
    if Lx_estimate is None:
        max_jump = np.max(np.abs(dtilt))
        if max_jump < 1.0:
            return np.ones(n, dtype=bool)
        Lx_estimate = max_jump
    threshold = flip_threshold * Lx_estimate
    flip_indices = np.where(np.abs(dtilt) > threshold)[0]
    if len(flip_indices) == 0:
        return np.ones(n, dtype=bool)
    mask = np.ones(n, dtype=bool)
    for fi in flip_indices:
        lo = max(0, fi - flip_window + 1)
        hi = min(n, fi + flip_window + 1)
        mask[lo:hi] = False
    return mask

def average_last_fraction_flipaware(arr, tilt_xy, frac=0.30, Lx=None):
    n = len(arr)
    i0 = int((1.0 - frac) * n)
    arr_tail = arr[i0:]
    tilt_tail = tilt_xy[i0:]
    mask = get_flip_mask(tilt_tail, Lx_estimate=Lx)
    clean = arr_tail[mask]
    if len(clean) == 0:
        return np.nan
    return np.mean(clean)

def extract_chialvo(folder):
    file_path = os.path.join(folder, "chialvo_stats.dat")
    data = np.atleast_2d(np.loadtxt(file_path))
    has_tilt = data.shape[1] >= 6
    pScaled    = data[:, 1]
    tau_over_p = data[:, 2]
    TScaled    = data[:, 3]
    if has_tilt:
        tilt_xy = data[:, 5]
        Lx = None
        out = {
            "pScaled":    average_last_fraction_flipaware(pScaled, tilt_xy, steady_fraction, Lx),
            "tau_over_p": average_last_fraction_flipaware(tau_over_p, tilt_xy, steady_fraction, Lx),
            "TScaled":    average_last_fraction_flipaware(TScaled, tilt_xy, steady_fraction, Lx),
        }
    else:
        n = len(pScaled)
        i0 = int((1.0 - steady_fraction) * n)
        out = {
            "pScaled":    np.mean(pScaled[i0:]),
            "tau_over_p": np.mean(tau_over_p[i0:]),
            "TScaled":    np.mean(TScaled[i0:]),
        }

    # Use ratio of means for tau/p to reduce bias from averaging an
    # instantaneous ratio in noisy/high-dissipation cases.
    raw_path = os.path.join(folder, "raw_stress.dat")
    if os.path.exists(raw_path):
        try:
            raw = np.atleast_2d(np.loadtxt(raw_path))
            if raw.shape[1] >= 7:
                p_raw = raw[:, 5]
                tau_raw = raw[:, 6]
                if raw.shape[1] >= 8:
                    tilt_raw = raw[:, 7]
                    p_mean = average_last_fraction_flipaware(p_raw, tilt_raw, steady_fraction, None)
                    tau_mean = average_last_fraction_flipaware(tau_raw, tilt_raw, steady_fraction, None)
                else:
                    n = len(p_raw)
                    i0 = int((1.0 - steady_fraction) * n)
                    p_mean = np.mean(p_raw[i0:])
                    tau_mean = np.mean(tau_raw[i0:])
                if np.isfinite(p_mean) and np.isfinite(tau_mean) and p_mean != 0.0:
                    out["tau_over_p"] = tau_mean / p_mean
        except Exception:
            pass

    return out

# =============================================================
# Collect DEM results
# =============================================================
results = {}

for e_case in e_cases:
    e_val = float(e_case.split("_")[1]) / 100.0
    results[e_val] = {"phi": [], "pScaled": [], "tau_over_p": [], "TScaled": []}

    for phi_case in phi_cases:
        phi_val = float(phi_case.split("_")[1]) / 100.0
        folder = os.path.join(base_dir, e_case, phi_case)
        if not os.path.exists(folder):
            continue
        try:
            out = extract_chialvo(folder)
            results[e_val]["phi"].append(phi_val)
            results[e_val]["pScaled"].append(out["pScaled"])
            results[e_val]["tau_over_p"].append(out["tau_over_p"])
            results[e_val]["TScaled"].append(out["TScaled"])
        except Exception as ex:
            print(f"Skipping {folder}: {ex}")

# Convert to arrays
for e_val in results:
    for k in results[e_val]:
        results[e_val][k] = np.array(results[e_val][k])

# =============================================================
# GD theory curves
# =============================================================
phi_theory = np.linspace(0.005, 0.63, 500)
e_values = sorted(results.keys(), reverse=True)  # [0.99, 0.95, 0.90, 0.80, 0.70]

# Marker/color assignments matching Chialvo's style
markers = {0.99: 'o', 0.95: 's', 0.90: '^', 0.80: 'v', 0.70: 'D'}
colors  = {0.99: 'C0', 0.95: 'C1', 0.90: 'C2', 0.80: 'C3', 0.70: 'C4'}

phi_c_mu0 = 0.636

# =============================================================
# FIGURE (a): Scaled pressure vs φ
# =============================================================
fig, ax = plt.subplots(figsize=(7, 5.5))

for e_val in e_values:
    gd = gd_steady_state(phi_theory, e_val, phi_c=phi_c_mu0)
    ax.semilogy(phi_theory, gd['pScaled'], '-', color=colors[e_val], lw=1.5, alpha=0.8)

    r = results[e_val]
    order = np.argsort(r['phi'])
    ax.semilogy(r['phi'][order], r['pScaled'][order],
                markers[e_val], color=colors[e_val], ms=7, mew=1.2,
                mfc='none', label=f'e={e_val:.2f}')

ax.set_xlabel(r'$\phi$', fontsize=13)
ax.set_ylabel(r'$p\,/\,\rho_s(\dot{\gamma}d)^2$', fontsize=13)
ax.set_title(r'Scaled Pressure — DEM vs GD Theory ($\mu=0$)', fontsize=12)
ax.legend(fontsize=10, ncol=2)
ax.set_xlim(-0.02, 0.65)
ax.set_ylim(1e-2, 1e5)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'pScaled_vs_phi_GD.png'), dpi=180)

# =============================================================
# FIGURE (b): Shear stress ratio τ/p vs φ
# =============================================================
fig, ax = plt.subplots(figsize=(7, 5.5))

for e_val in e_values:
    gd = gd_steady_state(phi_theory, e_val, phi_c=phi_c_mu0)
    ax.plot(phi_theory, gd['tau_over_p'], '-', color=colors[e_val], lw=1.5, alpha=0.8)

    r = results[e_val]
    order = np.argsort(r['phi'])
    ax.plot(r['phi'][order], r['tau_over_p'][order],
            markers[e_val], color=colors[e_val], ms=7, mew=1.2,
            mfc='none', label=f'e={e_val:.2f}')

ax.set_xlabel(r'$\phi$', fontsize=13)
ax.set_ylabel(r'$\tau\,/\,p$', fontsize=13)
ax.set_title(r'Shear Stress Ratio — DEM vs GD Theory ($\mu=0$)', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-0.02, 0.65)
ax.set_ylim(0, 0.65)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'tau_over_p_vs_phi_GD.png'), dpi=180)

# =============================================================
# FIGURE (c): Scaled temperature vs φ
# =============================================================
fig, ax = plt.subplots(figsize=(7, 5.5))

for e_val in e_values:
    gd = gd_steady_state(phi_theory, e_val, phi_c=phi_c_mu0)
    ax.semilogy(phi_theory, gd['TScaled'], '-', color=colors[e_val], lw=1.5, alpha=0.8)

    r = results[e_val]
    order = np.argsort(r['phi'])
    ax.semilogy(r['phi'][order], r['TScaled'][order],
                markers[e_val], color=colors[e_val], ms=7, mew=1.2,
                mfc='none', label=f'e={e_val:.2f}')

ax.set_xlabel(r'$\phi$', fontsize=13)
ax.set_ylabel(r'$T\,/\,(\dot{\gamma}d)^2$', fontsize=13)
ax.set_title(r'Scaled Temperature — DEM vs GD Theory ($\mu=0$)', fontsize=12)
ax.legend(fontsize=10, ncol=2)
ax.set_xlim(-0.02, 0.65)
ax.set_ylim(1e-1, 2e4)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'TScaled_vs_phi_GD.png'), dpi=180)

print("Saved: pScaled_vs_phi_GD.png, tau_over_p_vs_phi_GD.png, TScaled_vs_phi_GD.png")
print("\nDone. Lines = GD theory, open symbols = your DEM data.")
