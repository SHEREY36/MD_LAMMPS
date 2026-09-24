"""Homogeneous DSMC for dilute inelastic hard spheres in USF (Boltzmann limit).
Peculiar-frame free streaming: dc_x/dt = -k*a*c_y (k=1 physical, k=2 = what the
LAMMPS addforce produces). Units: sigma=1, m=1, a=1. n chosen to match phi=0.01."""
import numpy as np, sys
def run(alpha, k=1.0, N=40000, phi=0.01, a=1.0, dt=0.01, tmax=120.0, seed=1, T0=200.0):
    rng = np.random.default_rng(seed)
    n = 6*phi/np.pi                      # number density for sigma=1
    c = rng.normal(0, np.sqrt(T0), (N, 3)); c -= c.mean(0)
    gmax = 6*np.sqrt(T0); rows = []
    for it in range(int(tmax/dt)):
        c[:, 0] -= k*a*c[:, 1]*dt        # exact streaming step in peculiar frame
        # NTC collisions on disjoint random pairs
        Mexp = 0.5*N*n*np.pi*gmax*dt
        M = int(Mexp) + (rng.random() < Mexp-int(Mexp))
        M = min(M, N//2)
        p = rng.permutation(N)[:2*M].reshape(M, 2)
        g = c[p[:, 0]]-c[p[:, 1]]; gm = np.linalg.norm(g, axis=1)
        gmax = max(gmax, gm.max())
        acc = rng.random(M) < gm/gmax
        i, j, g, gm = p[acc, 0], p[acc, 1], g[acc], gm[acc]
        # impact direction k: P(k) ∝ (g_hat.k) on hemisphere -> cos = sqrt(U)
        gh = g/gm[:, None]
        cth = np.sqrt(rng.random(len(gm))); sth = np.sqrt(1-cth**2); ph = 2*np.pi*rng.random(len(gm))
        tmp = np.where(np.abs(gh[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = np.cross(gh, tmp); e1 /= np.linalg.norm(e1, axis=1)[:, None]; e2 = np.cross(gh, e1)
        kv = cth[:, None]*gh + (sth*np.cos(ph))[:, None]*e1 + (sth*np.sin(ph))[:, None]*e2
        gn = np.sum(g*kv, axis=1)
        dv = 0.5*(1+alpha)*gn[:, None]*kv
        c[i] -= dv; c[j] += dv
        if it % 20 == 0:
            P = (c.T @ c)/N              # P_ij/n  (m=1)
            T = np.trace(P)/3
            rows.append((it*dt, T, P[0, 0]/T, P[1, 1]/T, P[2, 2]/T, P[0, 1]/T))
    return np.array(rows)
if __name__ == "__main__":
    for alpha in (0.5, 0.7, 0.9):
        for k in (1.0, 2.0):
            r = run(alpha, k)
            s = r[r[:, 0] > 60]
            T = s[:, 1].mean()
            print(f"alpha={alpha} k={k:.0f}  T/(m a^2 d^2)={T:8.2f}  x(pi/6)={T*np.pi/6:8.2f}  "
                  f"Pxx*={s[:,2].mean():.4f} Pyy*={s[:,3].mean():.4f} Pzz*={s[:,4].mean():.4f} Pxy*={s[:,5].mean():.4f}", flush=True)
