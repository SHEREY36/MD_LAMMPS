#!/usr/bin/env python3
"""
Generate a LAMMPS data file for spherocylinder HCS, mirroring the
MFIX Fortran particle generator (particles.f90).

Particles are placed randomly (no lattice) with explicit
spherocylinder overlap checking, and velocities are drawn from
Maxwell-Boltzmann distributions with zero bulk momentum.

The output is a LAMMPS 'read_data' compatible file using atom_style ellipsoid.

Usage:
    python3 gen_spherocyl_lammps.py [--phi 0.01] [--AR 2.0] [--d 1.0]
                                     [--rho 0.763944] [--T_tr 1.0] [--T_rot 1.0]
                                     [--seed 42] [--outfile spherocyl_hcs.data]
"""

import numpy as np
import argparse
import sys

# ──────────────────────────────────────────────────────────────
# Spherocylinder segment–segment closest-approach distance²
# (same algorithm as particles.f90 / LAMMPS closest_approach)
# ──────────────────────────────────────────────────────────────
def seg_seg_distsq(r1, r2, u1, u2, hl):
    """
    Squared distance between two spherocylinder centerline segments.
    r1, r2: center positions (3,)
    u1, u2: unit orientation vectors (3,)
    hl: half cylinder length (same for both, monodisperse)
    """
    r21 = r2 - r1
    u1dot = np.dot(u1, r21)
    u2dot = np.dot(u2, r21)
    udot = np.dot(u1, u2)
    denom = 1.0 - udot**2

    SMALL = 1e-15
    if denom < SMALL:
        # Nearly parallel
        r21sq = np.dot(r21, r21)
        distsq = r21sq - u1dot**2 + max(0.0, abs(u1dot) - 2.0*hl)**2
        return distsq

    odenom = 1.0 / denom
    lam = (u1dot - udot * u2dot) * odenom
    mu = (-u2dot + udot * u1dot) * odenom

    v1 = abs(lam) - hl
    v2 = abs(mu) - hl

    if v1 > 0.0 or v2 > 0.0:
        if v1 > v2:
            lam = np.copysign(hl, lam)
            mu = lam * udot - u2dot
            if abs(mu) > hl:
                mu = np.copysign(hl, mu)
        else:
            mu = np.copysign(hl, mu)
            lam = mu * udot + u1dot
            if abs(lam) > hl:
                lam = np.copysign(hl, lam)

    r21sq = np.dot(r21, r21)
    distsq = (r21sq + lam**2 + mu**2
              - 2.0*lam*mu*udot + 2.0*mu*u2dot - 2.0*lam*u1dot)
    return max(distsq, 0.0)


# ──────────────────────────────────────────────────────────────
# Random uniform orientation on the sphere
# ──────────────────────────────────────────────────────────────
def random_orientation(rng):
    phi = np.arccos(rng.uniform(-1.0, 1.0))
    theta = rng.uniform(0.0, 2.0*np.pi)
    return np.array([np.sin(phi)*np.cos(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(phi)])


# ──────────────────────────────────────────────────────────────
# Orientation → quaternion (axis = unit vector u, treated as
# rotation from reference z-axis to u)
# ──────────────────────────────────────────────────────────────
def orient_to_quat(u):
    """
    Quaternion that rotates the body z-axis [0,0,1] to u.
    LAMMPS convention: quat = (w, x, y, z) stored as (w, i, j, k).
    """
    z = np.array([0.0, 0.0, 1.0])
    d = np.dot(z, u)
    if d > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -1.0 + 1e-12:
        # 180 degree rotation about any perpendicular axis
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, u)
    axis /= np.linalg.norm(axis)
    half_angle = 0.5 * np.arccos(np.clip(d, -1.0, 1.0))
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis
    return np.array([w, xyz[0], xyz[1], xyz[2]])


# ──────────────────────────────────────────────────────────────
# Rejection-sampled Gaussian (mimics the Fortran generator)
# ──────────────────────────────────────────────────────────────
def rejection_gaussian(rng, std, n=1, vmax=5.0):
    """Draw n samples from N(0, std) via acceptance-rejection on [-vmax, vmax]."""
    samples = []
    while len(samples) < n:
        vs = rng.uniform(-vmax, vmax)
        pv = np.exp(-0.5 * vs**2)
        if rng.random() < pv:
            samples.append(vs * std)
    return np.array(samples)


# ──────────────────────────────────────────────────────────────
# Spherocylinder principal moments of inertia
# ──────────────────────────────────────────────────────────────
def spherocyl_moments(rho, d, lcyl):
    d2, d3, d4, d5 = d**2, d**3, d**4, d**5
    l2, l3 = lcyl**2, lcyl**3
    iperp = (np.pi/48)*rho*d2*l3 + (np.pi/24)*rho*d3*l2 + \
            (3*np.pi/64)*rho*d4*lcyl + (np.pi/60)*rho*d5
    ipar  = (np.pi/32)*rho*d4*lcyl + (np.pi/60)*rho*d5
    return iperp, ipar


# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate LAMMPS data for spherocylinder HCS")
    parser.add_argument("--phi",   type=float, default=0.01)
    parser.add_argument("--AR",    type=float, default=2.0)
    parser.add_argument("--d",     type=float, default=1.0)
    parser.add_argument("--rho",   type=float, default=0.763944)
    parser.add_argument("--T_tr",  type=float, default=1.0)
    parser.add_argument("--T_rot", type=float, default=1.0)
    parser.add_argument("--N",     type=int,   default=None,
                        help="Number of particles (default: computed from phi to ~2003)")
    parser.add_argument("--Lbox",  type=float, default=None,
                        help="Box side length (default: computed from N and phi)")
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--outfile", type=str, default="spherocyl_hcs.data")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Geometry
    d = args.d
    R = 0.5 * d
    Lcyl = (args.AR - 1.0) * d
    hl = 0.5 * Lcyl
    C = R + hl           # semi-axis along rod for ellipsoid shape
    Vcaps = (4.0/3.0) * np.pi * R**3
    Vcyl = np.pi * R**2 * Lcyl
    Vp = Vcaps + Vcyl
    mass = args.rho * Vp

    # Number of particles
    if args.N is not None:
        N = args.N
    elif args.Lbox is not None:
        Vbox = args.Lbox**3
        N = int(np.ceil(Vbox * args.phi / Vp))
    else:
        # Match MFIX: compute N from phi and a box size that gives ~2003
        # MFIX uses L=64 with phi giving N=ceil(64^3 * phi / Vp)
        # Let's compute N the MFIX way using L=64 (in LJ units, same as MFIX's SI)
        # But actually the user may want to specify. Default: aim for ~2003.
        # We solve: N = ceil(L^3 * phi / Vp). With MFIX L=64 in SI with d=1:
        # Use the approach: pick N ~ 2003, then compute L from phi.
        N = int(np.ceil(64.0**3 * args.phi / Vp))  # should give ~2003 for the MFIX params

    # Box size
    if args.Lbox is not None:
        L = args.Lbox
    else:
        Vbox = N * Vp / args.phi
        L = Vbox ** (1.0/3.0)

    # Actual volume fraction
    phi_actual = N * Vp / L**3

    # Moments of inertia
    Iperp, Ipar = spherocyl_moments(args.rho, d, Lcyl)

    print(f"Spherocylinder HCS particle generator")
    print(f"  AR = {args.AR}, d = {d}, R = {R}, Lcyl = {Lcyl}, H = {hl}")
    print(f"  Vp = {Vp:.6f}, mass = {mass:.6f}")
    print(f"  N = {N}, L = {L:.4f}")
    print(f"  phi_target = {args.phi}, phi_actual = {phi_actual:.6e}")
    print(f"  Iperp = {Iperp:.6e}, Ipar = {Ipar:.6e}")
    print(f"  T_tr = {args.T_tr}, T_rot = {args.T_rot}")

    # ── Place particles randomly with overlap rejection ──
    # Exclusion distance: center-to-center segment distance must be > d
    # Buffer from walls for periodic wrapping: place anywhere in [0, L)
    # (periodic, so no wall buffer needed)
    bf = R + hl  # bounding half-length

    positions = np.zeros((N, 3))
    orientations = np.zeros((N, 3))  # unit orientation vectors

    print(f"Placing {N} particles with overlap rejection...")
    placed = 0
    max_attempts = 0
    while placed < N:
        # Random orientation
        u_new = random_orientation(rng)

        # Random position in [0, L)
        pos_new = rng.uniform(0.0, L, size=3)

        # Check overlaps with all previously placed particles
        # (with minimum image convention for periodic box)
        ok = True
        for j in range(placed):
            dr = pos_new - positions[j]
            # Minimum image
            dr -= L * np.round(dr / L)
            # Temporarily shift to check segment distance with shifted coords
            r1 = np.zeros(3)
            r2 = dr
            dsq = seg_seg_distsq(r1, r2, u_new, orientations[j], hl)
            if dsq < d**2:
                ok = False
                break

        if ok:
            positions[placed] = pos_new
            orientations[placed] = u_new
            placed += 1
            if placed % 500 == 0:
                print(f"  placed {placed}/{N}")

    print(f"  All {N} particles placed successfully.")

    # ── Generate velocities (Maxwell-Boltzmann, rejection sampling) ──
    vel_std_tr = np.sqrt(args.T_tr / mass)  # std for each component
    vel = np.zeros((N, 3))
    for i in range(N):
        for k in range(3):
            vel[i, k] = rejection_gaussian(rng, vel_std_tr, n=1)[0]

    # Remove bulk momentum
    vel -= vel.mean(axis=0)

    # ── Generate angular momenta ──
    # Smooth spherocylinder: 2 rotational DOF (perpendicular to rod axis)
    # T_rot = <L_perp^2 / I_perp> / 2  (per DOF, 2 DOFs)
    # => T_rot = <L_perp1^2>/(I_perp) = <L_perp2^2>/(I_perp)
    # => std(L_perp_component) = sqrt(T_rot * I_perp)
    # Angular momentum is in the SPACE frame (LAMMPS convention).
    # We generate 2 components in the body-perpendicular plane, then
    # convert to space-frame angular momentum.

    angmom_std = np.sqrt(args.T_rot * Iperp)
    angmom = np.zeros((N, 3))

    for i in range(N):
        u_rod = orientations[i]

        # Build a perpendicular basis
        if abs(u_rod[0]) < 0.9:
            perp1 = np.cross(u_rod, np.array([1.0, 0.0, 0.0]))
        else:
            perp1 = np.cross(u_rod, np.array([0.0, 1.0, 0.0]))
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(u_rod, perp1)
        perp2 /= np.linalg.norm(perp2)

        # Two perpendicular angular momentum components
        L1 = rejection_gaussian(rng, angmom_std, n=1)[0]
        L2 = rejection_gaussian(rng, angmom_std, n=1)[0]

        angmom[i] = L1 * perp1 + L2 * perp2
        # No axial spin component (smooth rod)

    # ── Convert orientations to quaternions ──
    quats = np.zeros((N, 4))
    for i in range(N):
        quats[i] = orient_to_quat(orientations[i])

    # ── Write LAMMPS data file ──
    # atom_style ellipsoid format:
    #   Atoms section: atom-ID atom-type x y z
    #   (with ellipsoid bonus data for shape and quaternion)
    #
    # The box is [0, L] x [0, L] x [0, L]

    with open(args.outfile, 'w') as f:
        f.write(f"LAMMPS data file for spherocylinder HCS (N={N}, phi={phi_actual:.6e}, AR={args.AR})\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"1 atom types\n")
        f.write(f"\n")
        f.write(f"0.0 {L:.10f} xlo xhi\n")
        f.write(f"0.0 {L:.10f} ylo yhi\n")
        f.write(f"0.0 {L:.10f} zlo zhi\n")
        f.write(f"\n")

        # Masses
        f.write(f"Masses\n\n")
        f.write(f"1 {mass:.10e}\n")
        f.write(f"\n")

        # Atoms section (atom_style ellipsoid):
        # atom-ID type x y z
        # The ellipsoidflag and density are set via Ellipsoids section
        f.write(f"Atoms # ellipsoid\n\n")
        for i in range(N):
            # atom-ID  type  x  y  z  ellipsoidflag  density
            f.write(f"{i+1} 1 {positions[i,0]:.10e} {positions[i,1]:.10e} {positions[i,2]:.10e} 1 {args.rho:.10e}\n")
        f.write(f"\n")

        # Ellipsoids section: atom-ID  shapex shapey shapez  quatw quati quatj quatk
        # For spherocylinder stored as ellipsoid: shape = (R, R, C) where C = R + H
        f.write(f"Ellipsoids\n\n")
        for i in range(N):
            q = quats[i]
            f.write(f"{i+1} {R:.10e} {R:.10e} {C:.10e} {q[0]:.10e} {q[1]:.10e} {q[2]:.10e} {q[3]:.10e}\n")
        f.write(f"\n")

        # Velocities section: atom-ID vx vy vz
        f.write(f"Velocities\n\n")
        for i in range(N):
            f.write(f"{i+1} {vel[i,0]:.10e} {vel[i,1]:.10e} {vel[i,2]:.10e} 0.0 0.0 0.0\n")
        f.write(f"\n")

        # AngularMomenta section: atom-ID Lx Ly Lz
        # (LAMMPS read_data for atom_style ellipsoid reads this from Velocities
        #  as columns 5,6,7: vx vy vz Lx Ly Lz)
        # Actually, for atom_style ellipsoid, Velocities has 6 columns:
        #   atom-ID vx vy vz Lx Ly Lz
        # So we need to go back and fix the Velocities section.

    # Write final data file with correct LAMMPS format for atom_style ellipsoid
    # (LAMMPS 2022+ new-style: fields_data_atom = {"id","type","ellipsoid","rmass","x"})
    with open(args.outfile, 'w') as f:
        f.write(f"LAMMPS data file for spherocylinder HCS (N={N}, phi={phi_actual:.6e}, AR={args.AR})\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"{N} ellipsoids\n")   # required for atom_style ellipsoid
        f.write(f"1 atom types\n")
        f.write(f"\n")
        f.write(f"0.0 {L:.10f} xlo xhi\n")
        f.write(f"0.0 {L:.10f} ylo yhi\n")
        f.write(f"0.0 {L:.10f} zlo zhi\n")
        f.write(f"\n")

        # atom_style ellipsoid Atoms format (new LAMMPS):
        #   atom-ID  type  ellipsoidflag  density  x  y  z
        # Note: ellipsoidflag and density come BEFORE x y z.
        # The per-atom mass is computed as density * (4/3*pi*a*b*c) using
        # semi-axes from the Ellipsoids section.  We override it afterward
        # with "set type 1 mass <m>" in the LAMMPS input script.
        f.write(f"Atoms # ellipsoid\n\n")
        for i in range(N):
            f.write(f"{i+1} 1 1 {args.rho:.10e} {positions[i,0]:.10e} {positions[i,1]:.10e} {positions[i,2]:.10e}\n")
        f.write(f"\n")

        # Ellipsoids section: atom-ID  shapex  shapey  shapez  qw  qi  qj  qk
        # LAMMPS reads shape as semi-axes = 0.5 * file_value (it expects diameters).
        # Write diameters (2R, 2R, 2C) so stored semi-axes are (R, R, C).
        f.write(f"Ellipsoids\n\n")
        for i in range(N):
            q = quats[i]
            f.write(f"{i+1} {2*R:.10e} {2*R:.10e} {2*C:.10e} {q[0]:.10e} {q[1]:.10e} {q[2]:.10e} {q[3]:.10e}\n")
        f.write(f"\n")

        # Velocities section for atom_style ellipsoid:
        #   atom-ID  vx  vy  vz  Lx  Ly  Lz
        f.write(f"Velocities\n\n")
        for i in range(N):
            f.write(f"{i+1} {vel[i,0]:.10e} {vel[i,1]:.10e} {vel[i,2]:.10e} "
                    f"{angmom[i,0]:.10e} {angmom[i,1]:.10e} {angmom[i,2]:.10e}\n")
        f.write(f"\n")

    print(f"\nWrote {args.outfile}")
    print(f"  Box: [0, {L:.4f}]^3")
    print(f"  N = {N}, phi = {phi_actual:.6e}")
    print(f"  mass = {mass:.6e}")
    print(f"  Use in LAMMPS with:")
    print(f"    atom_style ellipsoid")
    print(f"    read_data {args.outfile}")


if __name__ == "__main__":
    main()
