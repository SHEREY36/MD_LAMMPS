"""
Post-processing for HCS spherocylinder simulations.

Usage:
  python postprocess_hcs.py --mode A --file hcs_temperatures_A_e1.0.dat
  python postprocess_hcs.py --mode B --dir .   (reads all hcs_temperatures_B_e*.dat)
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import os
import re

# ============================================================
# Mode A: Jeans rotational relaxation (e = 1.0)
# ============================================================
def jeans_relaxation(t, T_eq, tau_relax, T_rot0):
    """
    Jeans continuum model for rotational relaxation:
      T_rot(t) = T_eq - (T_eq - T_rot0) * exp(-t / tau_relax)
    
    At e=1 (elastic), total energy is conserved, so T_tr and T_rot
    equilibrate to a common T_eq determined by DOF partitioning:
      T_eq = (3*T_tr0 + 2*T_rot0) / 5   for 3 trans + 2 rot DOF
    """
    return T_eq - (T_eq - T_rot0) * np.exp(-t / tau_relax)


def analyze_mode_A(filepath):
    """Fit Jeans relaxation to T_rot(t) for elastic (e=1) case."""
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:
        raise RuntimeError(
            "Mode A requires scipy (scipy.optimize.curve_fit). Install scipy or use --mode B."
        ) from exc

    data = np.loadtxt(filepath, comments='#')
    t, Ttr, Trot, ratio, Ttot = data.T

    # Expected equilibrium from energy conservation
    T_eq_expected = Ttot[0]  # (3*Ttr0 + 2*Trot0)/5, should be ~constant

    # Fit Jeans model to T_rot(t)
    # Initial guesses
    p0 = [T_eq_expected, t[-1] / 10.0, Trot[0]]
    try:
        popt, pcov = curve_fit(jeans_relaxation, t, Trot, p0=p0,
                               bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        T_eq_fit, tau_fit, T_rot0_fit = popt
        perr = np.sqrt(np.diag(pcov))
        print(f"Jeans relaxation fit:")
        print(f"  T_eq       = {T_eq_fit:.6f} ± {perr[0]:.6f}")
        print(f"  tau_relax  = {tau_fit:.6f} ± {perr[1]:.6f}")
        print(f"  T_rot(0)   = {T_rot0_fit:.6f} ± {perr[2]:.6f}")
        print(f"  T_eq (from energy conservation) = {T_eq_expected:.6f}")
    except RuntimeError:
        print("Curve fit failed. Plotting raw data only.")
        popt = None

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # T_tr and T_rot vs time
    ax = axes[0]
    ax.plot(t, Ttr, 'b-', alpha=0.6, label=r'$T_{\rm tr}$')
    ax.plot(t, Trot, 'r-', alpha=0.6, label=r'$T_{\rm rot}$')
    ax.axhline(T_eq_expected, color='k', ls='--', alpha=0.4, label=r'$T_{\rm eq}$ (energy)')
    if popt is not None:
        t_fine = np.linspace(t[0], t[-1], 500)
        ax.plot(t_fine, jeans_relaxation(t_fine, *popt), 'r--', lw=2,
                label=rf'Jeans fit ($\tau = {tau_fit:.3f}$)')
    ax.set_ylabel('Temperature')
    ax.set_title('Mode A: Rotational relaxation (e = 1.0)')
    ax.legend()
    ax.set_yscale('log')

    # Ratio vs time
    ax = axes[1]
    ax.plot(t, ratio, 'g-', alpha=0.7)
    ax.axhline(1.0, color='k', ls='--', alpha=0.4)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$T_{\rm rot} / T_{\rm tr}$')
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig('hcs_modeA_relaxation.png', dpi=150)
    plt.show()
    print("Saved: hcs_modeA_relaxation.png")


# ============================================================
# Mode B: Asymptotic T_tr/T_rot vs e
# ============================================================
TEMP_PRINT_EVERY = 200


def parse_e_from_modeB_filename(path):
    base = os.path.basename(path)
    e_str = base.replace('hcs_temperatures_B_e', '').replace('.dat', '')
    return float(e_str)


def load_modeB_files(directory):
    pattern = os.path.join(directory, '**', 'hcs_temperatures_B_e*.dat')
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        pattern = os.path.join(directory, 'hcs_temperatures_B_e*.dat')
        files = sorted(glob.glob(pattern))
    return files


def infer_dt_from_log(log_path):
    if not os.path.isfile(log_path):
        return None
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as fh:
        text = fh.read()
    matches = re.findall(r"Time step\s*:\s*([0-9eE+\-.]+)", text)
    if not matches:
        return None
    return float(matches[-1])


def infer_dt_from_temperature_times(t):
    if len(t) < 2:
        return None
    dt_sample = np.median(np.diff(t))
    if dt_sample <= 0.0:
        return None
    return float(dt_sample / TEMP_PRINT_EVERY)


def parse_npart_from_input(in_path):
    if not os.path.isfile(in_path):
        return None
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            m = re.match(r"\s*variable\s+N_part\s+equal\s+([0-9eE+\-.]+)", line)
            if m:
                return float(m.group(1))
    return None


def _load_collision_cpp_file(path):
    data = np.loadtxt(path, comments='#')
    data = np.atleast_2d(data)
    if data.shape[1] < 3 or len(data) < 2:
        return None
    t = data[:, 0]
    cpp = data[:, 1]
    rate = data[:, 2]
    return {
        "t": t,
        "cpp": cpp,
        "rate": rate,
        "source": os.path.basename(path),
    }


def load_collision_series(case_dir, t_temp):
    # Preferred: already converted file written by input script
    candidates = sorted(glob.glob(os.path.join(case_dir, "collision_cpp_B_e*.dat")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(case_dir, "collision_cpp_*.dat")))
    for path in candidates:
        coll = _load_collision_cpp_file(path)
        if coll is not None:
            return coll

    # Fallback: reconstruct from collision_events.dat
    events_path = os.path.join(case_dir, "collision_events.dat")
    if not os.path.isfile(events_path):
        return None

    data = np.loadtxt(events_path, comments='#')
    data = np.atleast_2d(data)
    if data.shape[1] < 3 or len(data) < 2:
        return None

    dt = infer_dt_from_log(os.path.join(case_dir, "log.lammps"))
    if dt is None:
        dt = infer_dt_from_temperature_times(t_temp)
    n_part = parse_npart_from_input(os.path.join(case_dir, "in.hcs"))

    if dt is None or n_part is None or n_part <= 0.0:
        return None

    step = data[:, 0]
    cum_events = data[:, 2]
    t = step * dt
    cpp = cum_events / n_part
    rate = np.zeros_like(cpp)
    dt_local = np.diff(t)
    dcpp = np.diff(cpp)
    rate[1:] = dcpp / (dt_local + 1.0e-30)
    return {
        "t": t,
        "cpp": cpp,
        "rate": rate,
        "source": "collision_events.dat (reconstructed)",
    }


def analyze_mode_B(directory, tail_frac=0.2):
    """
    Read all Mode B temperature files and generate:
      1) Haff-law style log-log overlay: T_total/T_total(0) vs time
      2) T_tr/T_rot vs time overlay
      3) Asymptotic mean T_tr/T_rot vs e
    """
    files = load_modeB_files(directory)
    if not files:
        print(f"No Mode B files found in {directory}")
        return

    # Parse and sort by restitution coefficient
    series = []
    for path in files:
        try:
            e_val = parse_e_from_modeB_filename(path)
        except ValueError:
            print(f"  Skipping {path}: cannot parse e")
            continue
        data = np.loadtxt(path, comments='#')
        data = np.atleast_2d(data)
        if len(data) < 20:
            print(f"  Skipping {path}: insufficient data")
            continue
        t, Ttr, Trot, ratio_rot_over_tr, Ttot = data.T
        ratio_tr_over_rot = Ttr / Trot
        case_dir = os.path.dirname(path)
        collision = load_collision_series(case_dir, t)
        series.append({
            "e": e_val,
            "t": t,
            "Ttr": Ttr,
            "Trot": Trot,
            "ratio": ratio_tr_over_rot,
            "Ttot": Ttot,
            "path": path,
            "collision": collision,
        })

    if not series:
        print("No valid Mode B datasets to process.")
        return

    series.sort(key=lambda x: x["e"])

    # -------------------------
    # Plot 1: Haff-law log-log overlay
    # -------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for s in series:
        e_val = s["e"]
        t = s["t"]
        Ttot = s["Ttot"]
        t0 = t[0]
        T0 = Ttot[0]
        y = Ttot / T0
        mask = (t > t0) & (y > 0.0)
        if np.count_nonzero(mask) < 2:
            continue
        ax1.loglog(t[mask], y[mask], lw=1.5, label=f"e={e_val:.2f}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"$T_{total}(t)/T_{total}(0)$")
    ax1.set_title("HCS Haff-Law Overlay (Mode B)")
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)
    fig1.tight_layout()
    fig1.savefig('hcs_modeB_haff_loglog.png', dpi=150)
    plt.close(fig1)
    print("Saved: hcs_modeB_haff_loglog.png")

    # -------------------------
    # Plot 2 + Plot 3 data prep
    # -------------------------
    summary = []
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for s in series:
        e_val = s["e"]
        t = s["t"]
        ratio_tr_over_rot = s["ratio"]
        ax2.plot(t, ratio_tr_over_rot, lw=1.3, label=f"e={e_val:.2f}")
        n_tail = max(10, int(len(t) * tail_frac))
        mean_tail = float(np.mean(ratio_tr_over_rot[-n_tail:]))
        std_tail = float(np.std(ratio_tr_over_rot[-n_tail:]))
        t_start_tail = float(t[-n_tail])
        t_end = float(t[-1])
        summary.append((e_val, mean_tail, std_tail, n_tail, t_start_tail, t_end))
        print(
            f"  e = {e_val:.3f}: T_tr/T_rot (tail mean) = {mean_tail:.6f} ± {std_tail:.6f} "
            f"(tail_frac={tail_frac:.2f})"
        )

    # Plot 2: T_tr/T_rot vs time overlay
    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"$T_{tr}/T_{rot}$")
    ax2.set_title(r"Mode B: $T_{tr}/T_{rot}$ vs Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)
    fig2.tight_layout()
    fig2.savefig('hcs_modeB_Ttr_over_Trot_vs_time.png', dpi=150)
    plt.close(fig2)
    print("Saved: hcs_modeB_Ttr_over_Trot_vs_time.png")

    # Plot 3: asymptotic mean T_tr/T_rot vs e
    summary = np.array(summary)
    e_arr = summary[:, 0]
    mean_arr = summary[:, 1]
    std_arr = summary[:, 2]
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.errorbar(e_arr, mean_arr, yerr=std_arr, fmt='o-', capsize=4,
                 color='steelblue', markersize=6)
    ax3.set_xlabel('Coefficient of restitution $e$', fontsize=12)
    ax3.set_ylabel(r'Asymptotic $T_{tr}/T_{rot}$', fontsize=12)
    ax3.set_title(f'HCS Spherocylinders (Mode B)')
    ax3.set_xlim(min(e_arr) - 0.02, max(e_arr) + 0.02)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig('hcs_modeB_asymptotic_Ttr_over_Trot_vs_e.png', dpi=150)
    plt.close(fig3)
    print("Saved: hcs_modeB_asymptotic_Ttr_over_Trot_vs_e.png")

    # Save table
    np.savetxt(
        'hcs_modeB_summary.dat',
        summary,
        header='e  Ttr_over_Trot_mean  std_dev  n_tail  tail_t_start  t_end',
        fmt='%.8g'
    )
    print("Saved: hcs_modeB_summary.dat")

    # -------------------------
    # Collision post-processing
    # -------------------------
    collision_series = [s for s in series if s["collision"] is not None]
    if not collision_series:
        print("No collision_cpp or collision_events data found. Skipping collision plots.")
        return

    # Plot 4: collisions per particle vs time
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    for s in collision_series:
        e_val = s["e"]
        coll = s["collision"]
        ax4.plot(coll["t"], coll["cpp"], lw=1.3, label=f"e={e_val:.2f}")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Collisions per particle")
    ax4.set_title("Mode B: Cumulative collisions per particle vs Time")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8, ncol=2)
    fig4.tight_layout()
    fig4.savefig('hcs_modeB_collisions_per_particle_vs_time.png', dpi=150)
    plt.close(fig4)
    print("Saved: hcs_modeB_collisions_per_particle_vs_time.png")

    # Plot 5 + Plot 6 data prep: collision rate
    coll_summary = []
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    for s in collision_series:
        e_val = s["e"]
        coll = s["collision"]
        t = coll["t"]
        rate = coll["rate"]
        ax5.plot(t, rate, lw=1.3, label=f"e={e_val:.2f}")

        n_tail = max(10, int(len(rate) * tail_frac))
        mean_tail = float(np.mean(rate[-n_tail:]))
        std_tail = float(np.std(rate[-n_tail:]))
        cpp_end = float(coll["cpp"][-1])
        t_start_tail = float(t[-n_tail])
        t_end = float(t[-1])
        coll_summary.append((e_val, cpp_end, mean_tail, std_tail, n_tail, t_start_tail, t_end))
        print(
            f"  e = {e_val:.3f}: collisions/particle(final) = {cpp_end:.6g}, "
            f"rate tail mean = {mean_tail:.6g} ± {std_tail:.6g} "
            f"(source: {coll['source']})"
        )

    ax5.set_xlabel("Time")
    ax5.set_ylabel("Collision rate per particle")
    ax5.set_title("Mode B: Collision rate per particle vs Time")
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8, ncol=2)
    fig5.tight_layout()
    fig5.savefig('hcs_modeB_collision_rate_vs_time.png', dpi=150)
    plt.close(fig5)
    print("Saved: hcs_modeB_collision_rate_vs_time.png")

    # Plot 6: asymptotic collision rate vs e
    coll_summary = np.array(coll_summary)
    e_arr = coll_summary[:, 0]
    rate_mean = coll_summary[:, 2]
    rate_std = coll_summary[:, 3]
    fig6, ax6 = plt.subplots(figsize=(7, 5))
    ax6.errorbar(e_arr, rate_mean, yerr=rate_std, fmt='o-', capsize=4,
                 color='darkorange', markersize=6)
    ax6.set_xlabel('Coefficient of restitution $e$', fontsize=12)
    ax6.set_ylabel('Asymptotic collision rate / particle', fontsize=12)
    ax6.set_title('HCS Spherocylinders (Mode B)')
    ax6.set_xlim(min(e_arr) - 0.02, max(e_arr) + 0.02)
    ax6.grid(True, alpha=0.3)
    fig6.tight_layout()
    fig6.savefig('hcs_modeB_asymptotic_collision_rate_vs_e.png', dpi=150)
    plt.close(fig6)
    print("Saved: hcs_modeB_asymptotic_collision_rate_vs_e.png")

    np.savetxt(
        'hcs_modeB_collision_summary.dat',
        coll_summary,
        header='e  cpp_final  coll_rate_mean  coll_rate_std  n_tail  tail_t_start  t_end',
        fmt='%.8g',
    )
    print("Saved: hcs_modeB_collision_summary.dat")


# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-process HCS temperature data')
    parser.add_argument('--mode', choices=['A', 'B'], required=True)
    parser.add_argument('--file', type=str, help='Temperature file (Mode A)')
    parser.add_argument('--dir', type=str, default='.', help='Directory with Mode B files')
    parser.add_argument('--tail-frac', type=float, default=0.2,
                        help='Tail fraction used for asymptotic averaging in Mode B')
    args = parser.parse_args()

    if args.mode == 'A':
        if not args.file:
            # Try to find it
            candidates = glob.glob('hcs_temperatures_A_e*.dat')
            if candidates:
                args.file = candidates[0]
            else:
                print("Provide --file for Mode A")
                exit(1)
        analyze_mode_A(args.file)
    else:
        if not (0.0 < args.tail_frac <= 1.0):
            raise ValueError("--tail-frac must be in (0, 1]")
        analyze_mode_B(args.dir, tail_frac=args.tail_frac)
