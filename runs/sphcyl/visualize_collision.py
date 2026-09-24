"""
visualize_collision.py
======================
Spherocylinder binary collision — animated validation plots for presentation.

Reads three output files:
  - log.lammps          (LAMMPS thermo: step, time, etotal, ke, ...)
  - contact_log.dat     (custom per-step: step time rcc sphere_gap seg_dist overlap)
  - sphcyl.dump         (atom dump: id type x y z vx vy vz angmomx angmomy angmomz quat[1-4])

Produces three animated GIFs (and matching static PNGs for PowerPoint):
  1. momentum_conservation.gif  — px, py, pz of each particle + total
  2. energy_dissipation.gif     — KE_trans + KE_rot vs time, analytical lines
  3. contact_geometry.gif       — rcc, overlap, sphere_gap vs time

REQUIRED DUMP COLUMNS (modify your LAMMPS dump line to):
  dump 1 all custom 10 sphcyl.dump id type x y z vx vy vz angmomx angmomy angmomz \\
       c_quat[1] c_quat[2] c_quat[3] c_quat[4]

If angmomx/y/z are not present the script still runs but KE_rot is zero.

Usage:
  python visualize_collision.py [--logfile log.lammps]
                                [--contactlog contact_log.dat]
                                [--dumpfile sphcyl.dump]
                                [--mass MASS]          # kg (auto-read if possible)
                                [--e RESTITUTION]      # e.g. 0.9
                                [--fps FPS]            # animation speed (default 15)
                                [--case LABEL]         # label for title e.g. "Case 3"

All arguments are optional; defaults match the reference LAMMPS script.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants from reference script (used if not auto-detected)
# ─────────────────────────────────────────────────────────────────────────────
D_DEFAULT   = 2.116e-4          # m
R_DEFAULT   = D_DEFAULT / 2.0
H_DEFAULT   = 2.0 * R_DEFAULT
RHO_DEFAULT = 2500.0            # kg/m³
Vsc_DEFAULT = (np.pi * R_DEFAULT**2 * (2*H_DEFAULT)
               + (4/3) * np.pi * R_DEFAULT**3)
M_DEFAULT   = RHO_DEFAULT * Vsc_DEFAULT

# Moment of inertia of a spherocylinder about a TRANSVERSE axis through CoM
# (the two rotational DOFs; used to compute KE_rot from angmom)
# I_trans = I_cylinder_transverse + I_hemisphere_transverse (Steiner)
# For a spherocylinder (cylinder of length 2H, radius R + two hemis):
#   I = m_cyl*(3R^2 + 4H^2)/12  + m_hemi*(2R^2/5 + H^2) * ... (full derivation below)
# We store I_transverse as a function of R, H, rho.
def moment_of_inertia_transverse(R, H, rho):
    """Moment of inertia about a transverse axis through CoM (the two equal I's)."""
    m_cyl  = rho * np.pi * R**2 * (2*H)
    m_hemi = rho * (2/3) * np.pi * R**3        # per hemisphere
    # Cylinder (length 2H, radius R) about transverse CoM axis:
    I_cyl = m_cyl * ((3*R**2 + (2*H)**2) / 12.0)
    # Each hemisphere (radius R) about its flat-face diameter axis:
    I_hemi_own = (83.0/320.0) * m_hemi * R**2
    # Distance from hemisphere CoM to rod CoM:
    d_hemi = H + (3*R/8.0)
    I_hemi = 2.0 * (I_hemi_own + m_hemi * d_hemi**2)  # both caps, Steiner
    return I_cyl + I_hemi

def moment_of_inertia_axial(R, H, rho):
    """Moment of inertia about the rod (symmetry) axis — for reference only."""
    m_cyl  = rho * np.pi * R**2 * (2*H)
    m_hemi = rho * (2/3) * np.pi * R**3
    I_cyl  = 0.5 * m_cyl * R**2
    I_hemi_axial = (2/5) * m_hemi * R**2
    return I_cyl + 2.0 * I_hemi_axial

# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_contact_log(filepath):
    """
    Parse contact_log.dat written by:
      fix clog all print 10 "$(step) $(time) $(v_rcc) $(v_sphere_gap) $(v_seg_dist) $(v_overlap)"
      title "step time rcc sphere_gap seg_dist overlap"
    Returns dict of numpy arrays keyed by column name.
    """
    path = Path(filepath)
    if not path.exists():
        warnings.warn(f"contact_log not found: {filepath}")
        return None

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("step"):
                continue
            try:
                vals = [float(v) for v in line.split()]
                if len(vals) >= 6:
                    rows.append(vals[:6])
            except ValueError:
                continue

    if not rows:
        return None

    arr = np.array(rows)
    return dict(
        step=arr[:,0], time=arr[:,1],
        rcc=arr[:,2], sphere_gap=arr[:,3],
        seg_dist=arr[:,4], overlap=arr[:,5]
    )


def parse_lammps_log(filepath):
    """
    Parse LAMMPS log for thermo output.
    Returns dict of column_name -> numpy array.
    Handles multiple thermo runs; concatenates all.
    """
    path = Path(filepath)
    if not path.exists():
        warnings.warn(f"log file not found: {filepath}")
        return None

    # Find header line(s) and data
    columns = None
    data_rows = []
    in_run = False

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("Step") or line.startswith("step"):
                # New thermo header
                columns = line.split()
                in_run = True
                continue
            if in_run:
                if re.match(r'^Loop|^WARNING|^ERROR|^LAMMPS', line):
                    in_run = False
                    continue
                try:
                    vals = [float(v) for v in line.split()]
                    if len(vals) == len(columns):
                        data_rows.append(vals)
                except (ValueError, TypeError):
                    pass

    if not data_rows or columns is None:
        return None

    arr = np.array(data_rows)
    return {col.lower(): arr[:,i] for i, col in enumerate(columns)}


def parse_dump(filepath):
    """
    Parse LAMMPS custom dump. Returns list of frames, each a dict:
      frame['timestep']  : int
      frame['natoms']    : int
      frame['columns']   : list of column names
      frame['data']      : ndarray (natoms, ncols)
    """
    path = Path(filepath)
    if not path.exists():
        warnings.warn(f"dump file not found: {filepath}")
        return []

    frames = []
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            ts = int(lines[i+1].strip())
            natoms = int(lines[i+3].strip())
            # columns line: "ITEM: ATOMS id type x y z ..."
            col_line = lines[i+8].strip()
            cols = col_line.replace("ITEM: ATOMS", "").split()
            # Normalise column names
            cols = [c.replace("c_quat[1]","quati")
                     .replace("c_quat[2]","quatj")
                     .replace("c_quat[3]","quatk")
                     .replace("c_quat[4]","quatw") for c in cols]
            atom_rows = []
            for j in range(natoms):
                atom_rows.append([float(v) for v in lines[i+9+j].split()])
            arr = np.array(atom_rows)
            # Sort by atom id
            id_col = cols.index("id") if "id" in cols else 0
            order = np.argsort(arr[:, id_col])
            frames.append(dict(timestep=ts, natoms=natoms,
                               columns=cols, data=arr[order]))
            i += 9 + natoms
        else:
            i += 1
    return frames


def extract_dump_series(frames, mass, R, H, rho):
    """
    From dump frames build time-series arrays.
    Returns dict with keys:
      time, px1, py1, pz1, px2, py2, pz2,
      ke_trans1, ke_trans2, ke_rot1, ke_rot2,
      ke_trans_total, ke_rot_total, ke_total,
      angmomx1, angmomy1, angmomz1, angmomx2, angmomy2, angmomz2
    """
    I_trans = moment_of_inertia_transverse(R, H, rho)
    I_axial = moment_of_inertia_axial(R, H, rho)

    timesteps = np.array([f['timestep'] for f in frames], dtype=float)
    # We don't know dt here; will be set by caller using contact log or log
    cols = frames[0]['columns']

    def col(name, data):
        if name in cols:
            return data[:, cols.index(name)]
        return np.zeros(data.shape[0])

    results = {k: [] for k in [
        'timestep',
        'px1','py1','pz1','px2','py2','pz2',
        'ke_trans1','ke_trans2','ke_rot1','ke_rot2',
        'angmomx1','angmomy1','angmomz1',
        'angmomx2','angmomy2','angmomz2',
    ]}

    for f in frames:
        d = f['data']   # shape (natoms, ncols)
        results['timestep'].append(f['timestep'])

        vx = col('vx', d); vy = col('vy', d); vz = col('vz', d)
        amx= col('angmomx', d); amy=col('angmomy', d); amz=col('angmomz', d)
        # LAMMPS angmom is L = I_body * omega in lab frame (body-frame scaled)
        # For KE_rot: KE = 0.5 * omega . L  but LAMMPS stores L in body frame
        # Simplest: KE_rot_i = 0.5 * (Lx^2 + Ly^2)/I_trans + 0.5*Lz^2/I_axial
        # Since rod is along body-Z, Lz is axial, Lx/Ly are transverse.
        # But we don't rotate L back to body frame here easily.
        # Use magnitude approximation: KE_rot ≈ |L|^2 / (2 * I_trans)
        # (valid when rotation is primarily transverse — true for smooth spherocylinders)
        L_sq1 = amx[0]**2 + amy[0]**2 + amz[0]**2
        L_sq2 = amx[1]**2 + amy[1]**2 + amz[1]**2

        results['px1'].append(mass * vx[0])
        results['py1'].append(mass * vy[0])
        results['pz1'].append(mass * vz[0])
        results['px2'].append(mass * vx[1])
        results['py2'].append(mass * vy[1])
        results['pz2'].append(mass * vz[1])

        results['ke_trans1'].append(0.5 * mass * (vx[0]**2 + vy[0]**2 + vz[0]**2))
        results['ke_trans2'].append(0.5 * mass * (vx[1]**2 + vy[1]**2 + vz[1]**2))
        results['ke_rot1'].append(0.5 * L_sq1 / I_trans)
        results['ke_rot2'].append(0.5 * L_sq2 / I_trans)

        results['angmomx1'].append(amx[0]); results['angmomy1'].append(amy[0]); results['angmomz1'].append(amz[0])
        results['angmomx2'].append(amx[1]); results['angmomy2'].append(amy[1]); results['angmomz2'].append(amz[1])

    return {k: np.array(v) for k, v in results.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'rod1': '#2196F3',    # blue
    'rod2': '#F44336',    # red
    'total': '#4CAF50',   # green
    'analytic': '#FF9800',# orange
    'contact': '#9C27B0', # purple
    'zero': '#888888',    # grey
}

def style_ax(ax, xlabel, ylabel, title, xlim=None, ylim=None):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Animation 1: Momentum conservation
# ─────────────────────────────────────────────────────────────────────────────

def make_momentum_animation(ts, data, contact_data, label, fps, outfile):
    """
    3-panel animation: px, py, pz vs time for rod1, rod2, total.
    A vertical band highlights the contact region (where overlap > 0).
    """
    # Time axis
    t = ts

    components = ['x', 'y', 'z']
    p1 = [data['px1'], data['py1'], data['pz1']]
    p2 = [data['px2'], data['py2'], data['pz2']]
    pt = [p1[i] + p2[i] for i in range(3)]

    # Contact window
    contact_mask = None
    if contact_data is not None:
        ov = contact_data['overlap']
        ct = contact_data['time']
        in_contact = ov > 0
        if np.any(in_contact):
            t_contact_start = ct[in_contact][0]
            t_contact_end   = ct[in_contact][-1]
        else:
            t_contact_start = t_contact_end = None
    else:
        t_contact_start = t_contact_end = None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f'Momentum Conservation — {label}', fontsize=14, fontweight='bold', y=1.02)

    ylim_all = []
    for i in range(3):
        all_vals = np.concatenate([p1[i], p2[i], pt[i]])
        pad = 0.15 * (all_vals.max() - all_vals.min() + 1e-40)
        ylim_all.append((all_vals.min() - pad, all_vals.max() + pad))

    lines_1, lines_2, lines_t = [], [], []
    shades = []

    for i, (ax, comp) in enumerate(zip(axes, components)):
        style_ax(ax,
                 xlabel='Time (s)',
                 ylabel=f'p{comp} (kg·m/s)',
                 title=f'{comp.upper()}-Momentum',
                 xlim=(t[0], t[-1]),
                 ylim=ylim_all[i])
        # Shade contact region
        if t_contact_start is not None:
            shade = ax.axvspan(t_contact_start, t_contact_end,
                               alpha=0.12, color=COLORS['contact'], label='Contact')
            shades.append(shade)
        # Analytic total (constant)
        p_total_val = pt[i][0]
        ax.axhline(p_total_val, color=COLORS['analytic'], lw=1.5,
                   linestyle='--', alpha=0.7, label='p_total (analytic)')
        l1, = ax.plot([], [], color=COLORS['rod1'], lw=2, label='Rod 1')
        l2, = ax.plot([], [], color=COLORS['rod2'], lw=2, label='Rod 2')
        lt, = ax.plot([], [], color=COLORS['total'], lw=2, ls=':', label='Total')
        lines_1.append(l1); lines_2.append(l2); lines_t.append(lt)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

    # Time cursor
    cursors = [ax.axvline(t[0], color='black', lw=1, ls=':', alpha=0.6)
               for ax in axes]

    # Annotation box (top left of first axis)
    step_text = axes[0].text(0.02, 0.96, '', transform=axes[0].transAxes,
                             fontsize=9, va='top', family='monospace',
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    fig.tight_layout()

    N = len(t)
    # Subsample for smooth animation (target ~150 frames)
    frame_indices = np.unique(np.linspace(0, N-1, min(N, 150)).astype(int))

    def init():
        for l1, l2, lt in zip(lines_1, lines_2, lines_t):
            l1.set_data([], []); l2.set_data([], []); lt.set_data([], [])
        return lines_1 + lines_2 + lines_t + cursors + [step_text]

    def update(frame_idx):
        k = frame_indices[frame_idx]
        tk = t[:k+1]
        for i in range(3):
            lines_1[i].set_data(tk, p1[i][:k+1])
            lines_2[i].set_data(tk, p2[i][:k+1])
            lines_t[i].set_data(tk, pt[i][:k+1])
            cursors[i].set_xdata([t[k], t[k]])
        # Conservation check: max deviation of total from initial
        dev = max(np.max(np.abs(pt[i][:k+1] - pt[i][0])) for i in range(3))
        step_text.set_text(f't = {t[k]:.2e} s\n|Δp_total|_max = {dev:.2e}')
        return lines_1 + lines_2 + lines_t + cursors + [step_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                  init_func=init, blit=True, interval=1000/fps)
    ani.save(outfile, writer='pillow', fps=fps, dpi=120)
    plt.savefig(outfile.replace('.gif', '_static.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outfile}  +  {outfile.replace('.gif','_static.png')}")


# ─────────────────────────────────────────────────────────────────────────────
# Animation 2: Energy dissipation
# ─────────────────────────────────────────────────────────────────────────────

def make_energy_animation(ts, data, contact_data, label, fps, outfile, e_rest, mass):
    """
    2-panel animation:
      Left:  KE_trans + KE_rot + KE_total vs time
             Overlaid: analytic initial/final KE lines
      Right: KE_rot only (both rods separately)
             Shows if rotational DOFs are activated

    Analytic:
      KE_initial = 0.5 * m * v0^2 * 2 particles
      KE_final   = KE_initial * e^2  (for head-on, equal-mass)
    """
    t = ts
    ke_trans = data['ke_trans1'] + data['ke_trans2']
    ke_rot   = data['ke_rot1']   + data['ke_rot2']
    ke_total = ke_trans + ke_rot

    ke_rot1  = data['ke_rot1']
    ke_rot2  = data['ke_rot2']

    KE_initial = ke_total[0]
    KE_final_analytic = KE_initial * e_rest**2

    # Contact window
    if contact_data is not None:
        ov = contact_data['overlap']
        ct = contact_data['time']
        in_contact = ov > 0
        t_cs = ct[in_contact][0]  if np.any(in_contact) else None
        t_ce = ct[in_contact][-1] if np.any(in_contact) else None
    else:
        t_cs = t_ce = None

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f'Kinetic Energy Dissipation — {label}', fontsize=14, fontweight='bold', y=1.02)

    # ---- Left: total KE breakdown ----
    ax = axes[0]
    pad_e = 0.15 * KE_initial
    style_ax(ax, 'Time (s)', 'Kinetic Energy (J)',
             'KE: Translational + Rotational',
             xlim=(t[0], t[-1]),
             ylim=(-pad_e*0.5, KE_initial + pad_e))

    if t_cs is not None:
        ax.axvspan(t_cs, t_ce, alpha=0.12, color=COLORS['contact'])

    # Analytic reference lines (static)
    ax.axhline(KE_initial,       color=COLORS['analytic'], lw=1.8, ls='--',
               label=f'KE₀ (analytic) = {KE_initial:.3e} J', alpha=0.85)
    ax.axhline(KE_final_analytic, color='brown', lw=1.8, ls='-.',
               label=f'KE_final = e²·KE₀ = {KE_final_analytic:.3e} J', alpha=0.85)
    ax.axhline(0, color=COLORS['zero'], lw=0.8, ls=':')

    l_trans, = ax.plot([], [], color=COLORS['rod1'], lw=2.2, label='KE_trans (both)')
    l_rot,   = ax.plot([], [], color=COLORS['rod2'], lw=2.2, ls='--', label='KE_rot (both)')
    l_tot,   = ax.plot([], [], color=COLORS['total'], lw=2.5, label='KE_total')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    cur_e = ax.axvline(t[0], color='black', lw=1, ls=':', alpha=0.6)

    # ---- Right: rotational KE per rod ----
    ax2 = axes[1]
    rot_max = max(ke_rot.max(), 1e-30)
    style_ax(ax2, 'Time (s)', 'KE_rot (J)',
             'Rotational Energy per Rod\n(0 = no rotation activated)',
             xlim=(t[0], t[-1]),
             ylim=(-0.05 * rot_max, rot_max * 1.2 + 1e-40))

    if t_cs is not None:
        ax2.axvspan(t_cs, t_ce, alpha=0.12, color=COLORS['contact'])
    ax2.axhline(0, color=COLORS['zero'], lw=1, ls='--', alpha=0.6,
                label='KE_rot = 0 (expected if centered)')

    lr1, = ax2.plot([], [], color=COLORS['rod1'], lw=2, label='KE_rot Rod 1')
    lr2, = ax2.plot([], [], color=COLORS['rod2'], lw=2, ls='--', label='KE_rot Rod 2')
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.8)
    cur_e2 = ax2.axvline(t[0], color='black', lw=1, ls=':', alpha=0.6)

    step_text = axes[0].text(0.02, 0.05, '', transform=axes[0].transAxes,
                             fontsize=9, va='bottom', family='monospace',
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    fig.tight_layout()

    N = len(t)
    frame_indices = np.unique(np.linspace(0, N-1, min(N, 150)).astype(int))

    def init():
        for l in [l_trans, l_rot, l_tot, lr1, lr2]:
            l.set_data([], [])
        return [l_trans, l_rot, l_tot, lr1, lr2, cur_e, cur_e2, step_text]

    def update(frame_idx):
        k = frame_indices[frame_idx]
        tk = t[:k+1]
        l_trans.set_data(tk, ke_trans[:k+1])
        l_rot.set_data(tk,   ke_rot[:k+1])
        l_tot.set_data(tk,   ke_total[:k+1])
        lr1.set_data(tk,     ke_rot1[:k+1])
        lr2.set_data(tk,     ke_rot2[:k+1])
        cur_e.set_xdata([t[k], t[k]])
        cur_e2.set_xdata([t[k], t[k]])
        ke_err = abs(ke_total[k] - KE_final_analytic) / (KE_initial + 1e-40) * 100
        step_text.set_text(f't = {t[k]:.2e} s\nKE_rot frac = {ke_rot[k]/(ke_total[k]+1e-40)*100:.2f}%\n|KE−KE_anal|/KE₀ = {ke_err:.2f}%')
        return [l_trans, l_rot, l_tot, lr1, lr2, cur_e, cur_e2, step_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                  init_func=init, blit=True, interval=1000/fps)
    ani.save(outfile, writer='pillow', fps=fps, dpi=120)
    plt.savefig(outfile.replace('.gif', '_static.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outfile}  +  {outfile.replace('.gif','_static.png')}")


# ─────────────────────────────────────────────────────────────────────────────
# Animation 3: Contact geometry proof
# ─────────────────────────────────────────────────────────────────────────────

def make_contact_animation(contact_data, label, fps, outfile, R, H):
    """
    2-panel animation:
      Left:  rcc vs time; marks sphere-sphere threshold (2R) and
             spherocylinder contact onset (~4R for T-collision)
      Right: overlap and sphere_gap vs time
             Rows where overlap>0 AND sphere_gap>0 prove segment detection.
    """
    if contact_data is None:
        print("  Skipping contact geometry plot (no contact_log.dat)")
        return

    ct   = contact_data['time']
    rcc  = contact_data['rcc']
    ov   = contact_data['overlap']
    sg   = contact_data['sphere_gap']
    sd   = contact_data['seg_dist']

    twoR  = 2.0 * R
    fourR = 4.0 * R

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f'Contact Geometry Proof — {label}', fontsize=14, fontweight='bold', y=1.02)

    # ---- Left: rcc ----
    ax = axes[0]
    rcc_pad = 0.15 * (rcc.max() - rcc.min() + 1e-30)
    style_ax(ax, 'Time (s)', 'Center-to-Center Distance (m)',
             'rcc vs Time',
             xlim=(ct[0], ct[-1]),
             ylim=(rcc.min() - rcc_pad, rcc.max() + rcc_pad))

    ax.axhline(twoR,  color='red',    lw=1.5, ls='--',
               label=f'2R = {twoR:.3e} m  (sphere threshold)')
    ax.axhline(fourR, color=COLORS['contact'], lw=1.5, ls='-.',
               label=f'4R = {fourR:.3e} m  (T-contact onset)')

    in_contact = ov > 0
    if np.any(in_contact):
        ax.axvspan(ct[in_contact][0], ct[in_contact][-1],
                   alpha=0.12, color=COLORS['contact'], label='Contact window')

    l_rcc, = ax.plot([], [], color='#1565C0', lw=2.5, label='rcc (LAMMPS)')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    cur1 = ax.axvline(ct[0], color='black', lw=1, ls=':', alpha=0.6)

    # ---- Right: overlap + sphere_gap ----
    ax2 = axes[1]
    all_vals = np.concatenate([ov, sg])
    pad2 = 0.2 * (all_vals.max() - all_vals.min() + 1e-30)
    style_ax(ax2, 'Time (s)', 'Distance (m)',
             'Overlap & Sphere-Gap\n(overlap>0 AND sphere_gap>0 → segment detection)',
             xlim=(ct[0], ct[-1]),
             ylim=(all_vals.min() - pad2, all_vals.max() + pad2))

    ax2.axhline(0, color=COLORS['zero'], lw=1, ls='--', alpha=0.7)
    if np.any(in_contact):
        ax2.axvspan(ct[in_contact][0], ct[in_contact][-1],
                    alpha=0.12, color=COLORS['contact'])

    l_ov, = ax2.plot([], [], color=COLORS['rod1'], lw=2.2, label='overlap  (>0 = in contact)')
    l_sg, = ax2.plot([], [], color=COLORS['rod2'], lw=2.2, ls='--',
                     label='sphere_gap (>0 = sphere-sphere blind)')
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.8)
    cur2 = ax2.axvline(ct[0], color='black', lw=1, ls=':', alpha=0.6)

    # Proof annotation (appears as soon as overlap>0 AND sphere_gap>0)
    proof_text = ax2.text(0.02, 0.96, '', transform=ax2.transAxes,
                          fontsize=9, va='top', color=COLORS['contact'], fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

    fig.tight_layout()

    N = len(ct)
    frame_indices = np.unique(np.linspace(0, N-1, min(N, 150)).astype(int))

    def init():
        l_rcc.set_data([], [])
        l_ov.set_data([], [])
        l_sg.set_data([], [])
        return [l_rcc, l_ov, l_sg, cur1, cur2, proof_text]

    def update(frame_idx):
        k = frame_indices[frame_idx]
        tk = ct[:k+1]
        l_rcc.set_data(tk, rcc[:k+1])
        l_ov.set_data(tk, ov[:k+1])
        l_sg.set_data(tk, sg[:k+1])
        cur1.set_xdata([ct[k], ct[k]])
        cur2.set_xdata([ct[k], ct[k]])
        # Proof text
        n_proof = np.sum((ov[:k+1] > 0) & (sg[:k+1] > 0))
        if n_proof > 0:
            proof_text.set_text(f'✓ SEGMENT DETECTION\n{n_proof} steps with\noverlap>0 AND sphere_gap>0')
            proof_text.set_color(COLORS['contact'])
        else:
            proof_text.set_text('')
        return [l_rcc, l_ov, l_sg, cur1, cur2, proof_text]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                  init_func=init, blit=True, interval=1000/fps)
    ani.save(outfile, writer='pillow', fps=fps, dpi=120)
    plt.savefig(outfile.replace('.gif', '_static.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outfile}  +  {outfile.replace('.gif','_static.png')}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary static figure (for PowerPoint slide)
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_figure(ts, data, contact_data, label, outfile, e_rest, mass, R, H):
    """
    One static 2×3 figure with all key physics for a single PowerPoint slide.
    Row 1: px, py, pz  (momentum conservation)
    Row 2: KE breakdown, KE_rot per rod, rcc geometry
    """
    t = ts
    p1 = [data['px1'], data['py1'], data['pz1']]
    p2 = [data['px2'], data['py2'], data['pz2']]
    pt = [p1[i]+p2[i] for i in range(3)]

    ke_trans = data['ke_trans1'] + data['ke_trans2']
    ke_rot   = data['ke_rot1']   + data['ke_rot2']
    ke_total = ke_trans + ke_rot
    KE0 = ke_total[0]
    KE_final = KE0 * e_rest**2

    if contact_data is not None:
        ov = contact_data['overlap']
        ct = contact_data['time']
        in_c = ov > 0
        t_cs = ct[in_c][0]  if np.any(in_c) else None
        t_ce = ct[in_c][-1] if np.any(in_c) else None
        rcc  = contact_data['rcc']
        sg   = contact_data['sphere_gap']
    else:
        t_cs = t_ce = None

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f'Spherocylinder Collision Validation — {label}\n'
                 f'e = {e_rest:.2f},  m = {mass:.3e} kg,  R = {R:.3e} m,  H = {H:.3e} m',
                 fontsize=13, fontweight='bold')

    comp_labels = ['x', 'y', 'z']
    for i in range(3):
        ax = axes[0, i]
        all_vals = np.concatenate([p1[i], p2[i], pt[i]])
        pad = 0.15 * (np.ptp(all_vals) + 1e-40)
        style_ax(ax, 'Time (s)', f'p{comp_labels[i]} (kg·m/s)',
                 f'{comp_labels[i].upper()}-Momentum Conservation')
        if t_cs is not None:
            ax.axvspan(t_cs, t_ce, alpha=0.12, color=COLORS['contact'])
        ax.axhline(pt[i][0], color=COLORS['analytic'], lw=1.5, ls='--', alpha=0.7,
                   label='p_total (conserved)')
        ax.plot(t, p1[i], color=COLORS['rod1'], lw=2, label='Rod 1')
        ax.plot(t, p2[i], color=COLORS['rod2'], lw=2, label='Rod 2')
        ax.plot(t, pt[i], color=COLORS['total'], lw=2, ls=':', label='Total')
        ax.set_ylim(all_vals.min()-pad, all_vals.max()+pad)
        if i == 0:
            ax.legend(fontsize=8, framealpha=0.8)

    # Row 2, col 0: KE breakdown
    ax = axes[1, 0]
    style_ax(ax, 'Time (s)', 'KE (J)', 'Kinetic Energy')
    if t_cs is not None:
        ax.axvspan(t_cs, t_ce, alpha=0.12, color=COLORS['contact'])
    ax.axhline(KE0,      color=COLORS['analytic'], lw=1.5, ls='--',
               label=f'KE₀ = {KE0:.3e} J', alpha=0.85)
    ax.axhline(KE_final, color='brown',            lw=1.5, ls='-.',
               label=f'e²·KE₀ = {KE_final:.3e} J', alpha=0.85)
    ax.plot(t, ke_trans, color=COLORS['rod1'], lw=2, label='KE_trans')
    ax.plot(t, ke_rot,   color=COLORS['rod2'], lw=2, ls='--', label='KE_rot')
    ax.plot(t, ke_total, color=COLORS['total'], lw=2.5, label='KE_total')
    ax.legend(fontsize=8, framealpha=0.8)

    # Row 2, col 1: KE_rot per rod
    ax = axes[1, 1]
    rot_max = max(data['ke_rot1'].max(), data['ke_rot2'].max(), 1e-40)
    style_ax(ax, 'Time (s)', 'KE_rot (J)',
             'Rotational Energy per Rod\n(0 = no spin activated)')
    if t_cs is not None:
        ax.axvspan(t_cs, t_ce, alpha=0.12, color=COLORS['contact'])
    ax.axhline(0, color=COLORS['zero'], lw=1, ls='--', alpha=0.6)
    ax.plot(t, data['ke_rot1'], color=COLORS['rod1'], lw=2, label='Rod 1 KE_rot')
    ax.plot(t, data['ke_rot2'], color=COLORS['rod2'], lw=2, ls='--', label='Rod 2 KE_rot')
    ax.set_ylim(-0.05*rot_max, rot_max*1.3 + 1e-40)
    ax.legend(fontsize=8, framealpha=0.8)

    # Row 2, col 2: rcc + overlap geometry
    ax = axes[1, 2]
    if contact_data is not None:
        style_ax(ax, 'Time (s)', 'm', 'Contact Geometry')
        ax.axhline(2*R, color='red', lw=1.2, ls='--',
                   label=f'2R sphere threshold')
        ax.axhline(4*R, color=COLORS['contact'], lw=1.2, ls='-.',
                   label=f'4R sphcyl T-onset')
        ax.plot(ct, rcc, color='#1565C0', lw=2, label='rcc')
        ax.plot(ct, sg,  color=COLORS['rod2'], lw=1.5, ls='--', label='sphere_gap')
        ax.plot(ct, contact_data['overlap'],
                color=COLORS['rod1'], lw=1.5, ls=':', label='overlap')
        ax.legend(fontsize=8, framealpha=0.8)
    else:
        axes[1, 2].set_visible(False)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved summary: {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Spherocylinder collision validation plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--logfile',    default='log.lammps')
    parser.add_argument('--contactlog', default='contact_log.dat')
    parser.add_argument('--dumpfile',   default='sphcyl.dump')
    parser.add_argument('--mass',       type=float, default=None,
                        help='Particle mass [kg]. Auto-computed if not given.')
    parser.add_argument('--R',          type=float, default=R_DEFAULT,
                        help='Particle radius [m]')
    parser.add_argument('--H',          type=float, default=H_DEFAULT,
                        help='Half-cylinder length [m]')
    parser.add_argument('--rho',        type=float, default=RHO_DEFAULT,
                        help='Particle density [kg/m3]')
    parser.add_argument('--e',          type=float, default=0.9,
                        help='Coefficient of restitution')
    parser.add_argument('--fps',        type=int,   default=15,
                        help='Animation frames per second')
    parser.add_argument('--case',       default='Binary Collision',
                        help='Case label for plot titles')
    parser.add_argument('--outdir',     default='.',
                        help='Output directory for GIFs and PNGs')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    R   = args.R
    H   = args.H
    rho = args.rho
    Vsc = np.pi*R**2*(2*H) + (4/3)*np.pi*R**3
    mass = args.mass if args.mass is not None else rho * Vsc
    e_rest = args.e

    print(f"\n=== Spherocylinder Validation Plots ===")
    print(f"  Case    : {args.case}")
    print(f"  R={R:.4e} m, H={H:.4e} m, mass={mass:.4e} kg, e={e_rest}")
    print(f"  Output  : {outdir}")
    print()

    # ── Parse data ──────────────────────────────────────────────────────────
    print("Parsing contact_log.dat ...")
    cdata = parse_contact_log(args.contactlog)

    print("Parsing log.lammps ...")
    ldata = parse_lammps_log(args.logfile)

    print("Parsing sphcyl.dump ...")
    frames = parse_dump(args.dumpfile)
    if not frames:
        print("ERROR: No frames found in dump file. Check path and format.")
        sys.exit(1)

    print(f"  Loaded {len(frames)} dump frames.")

    # ── Build time axis from dump timesteps ────────────────────────────────
    # Infer dt from log if possible; fallback to 1e-7
    dt = 1.0e-7
    if ldata is not None and 'time' in ldata and 'step' in ldata:
        steps = ldata['step']; times = ldata['time']
        if len(steps) > 1:
            dt = (times[-1] - times[0]) / (steps[-1] - steps[0] + 1e-40)

    dump_series = extract_dump_series(frames, mass, R, H, rho)
    ts = dump_series['timestep'] * dt   # convert step -> time

    # ── Check dump has angmom columns ─────────────────────────────────────
    has_angmom = 'angmomx' in frames[0]['columns']
    if not has_angmom:
        print("\n  WARNING: angmomx/y/z not found in dump. KE_rot will be zero.")
        print("  Add 'angmomx angmomy angmomz' to your dump line:")
        print("  dump 1 all custom 10 sphcyl.dump id type x y z vx vy vz \\")
        print("       angmomx angmomy angmomz c_quat[1] c_quat[2] c_quat[3] c_quat[4]")
        print()

    print("\nGenerating animations ...")

    # ── Animation 1: Momentum ──────────────────────────────────────────────
    out1 = str(outdir / 'momentum_conservation.gif')
    make_momentum_animation(ts, dump_series, cdata, args.case, args.fps, out1)

    # ── Animation 2: Energy ───────────────────────────────────────────────
    out2 = str(outdir / 'energy_dissipation.gif')
    make_energy_animation(ts, dump_series, cdata, args.case, args.fps, out2,
                          e_rest, mass)

    # ── Animation 3: Contact geometry ─────────────────────────────────────
    out3 = str(outdir / 'contact_geometry.gif')
    make_contact_animation(cdata, args.case, args.fps, out3, R, H)

    # ── Summary static PNG ─────────────────────────────────────────────────
    out4 = str(outdir / 'summary_validation.png')
    make_summary_figure(ts, dump_series, cdata, args.case, out4, e_rest, mass, R, H)

    print(f"""
=== Done ===
Output files in {outdir}:
  momentum_conservation.gif / _static.png   → slide 1 (beside OVITO)
  energy_dissipation.gif    / _static.png   → slide 2
  contact_geometry.gif      / _static.png   → slide 3
  summary_validation.png                     → 1-page summary slide

PowerPoint tip:
  Insert .gif files as animations (Insert > Pictures > This Device).
  They play automatically in slideshow mode.
  Use summary_validation.png for a static overview slide.
""")


if __name__ == '__main__':
    main()