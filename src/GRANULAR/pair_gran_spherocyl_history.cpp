#include "pair_gran_spherocyl_history.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "fix_dummy.h"
#include "fix_neigh_history.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "atom_vec_ellipsoid.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

PairGranSpherocylHistory::PairGranSpherocylHistory(LAMMPS *lmp)
    : PairGranHookeHistory(lmp)
{
  R = nullptr;
  H = nullptr;
}

PairGranSpherocylHistory::~PairGranSpherocylHistory()
{
  if (allocated) {
    memory->destroy(R);
    memory->destroy(H);
  }
}

void PairGranSpherocylHistory::allocate_sc()
{
  if (!allocated) PairGranHookeHistory::allocate();

  int n = atom->ntypes + 1;
  memory->create(R, n, "pair:gran/spherocyl:R");
  memory->create(H, n, "pair:gran/spherocyl:H");
  for (int i = 1; i <= atom->ntypes; i++) {
    R[i] = 0.0;
    H[i] = 0.0;
  }
}

void PairGranSpherocylHistory::coeff(int narg, char **arg)
{
  // Syntax:
  // pair_coeff  i j  R  H
  // For now, assign the same (R,H) to all types in the specified i and j ranges.
  // This is sufficient for monodisperse and most early testing.
  if (narg != 4)
    error->all(FLERR, "Incorrect args for pair coefficients for gran/spherocyl/history");

  allocate_sc();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double Rval = utils::numeric(FLERR, arg[2], false, lmp);
  double Hval = utils::numeric(FLERR, arg[3], false, lmp);

  if (Rval <= 0.0) error->all(FLERR, "gran/spherocyl/history: R must be > 0");
  if (Hval < 0.0)  error->all(FLERR, "gran/spherocyl/history: H must be >= 0");

  // set per-type geometry
  for (int t = ilo; t <= ihi; t++) { R[t] = Rval; H[t] = Hval; }
  for (int t = jlo; t <= jhi; t++) { R[t] = Rval; H[t] = Hval; }

  // enable type pairs
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }
  if (count == 0)
    error->all(FLERR, "Incorrect args for pair coefficients for gran/spherocyl/history");
}

void PairGranSpherocylHistory::init_style()
{
  // Do not rely on ellipsoid_flag/quat_flag here; they can be unset depending on init order.
  // We will do pointer checks in compute() before dereferencing.

  if (comm->ghost_velocity == 0)
    error->all(FLERR, "Pair gran/spherocyl/history requires ghost atoms store velocity (comm_modify vel yes)");

  // granular neighbor list with history
  if (history)
    neighbor->add_request(this, NeighConst::REQ_SIZE | NeighConst::REQ_HISTORY);
  else
    neighbor->add_request(this, NeighConst::REQ_SIZE);

  dt = update->dt;

  // Create/replace FixNeighHistory exactly like the parent does
  if (history && (fix_history == nullptr)) {
    // Use the same NEIGH_HISTORY_HH* naming as PairGranHookeHistory.
    // This avoids any mismatch/lookup problems and keeps behavior identical.
    auto cmd = fmt::format("NEIGH_HISTORY_HH{} all NEIGH_HISTORY {}", instance_me, size_history);
    fix_history = dynamic_cast<FixNeighHistory *>(
        modify->replace_fix("NEIGH_HISTORY_HH_DUMMY" + std::to_string(instance_me), cmd, 1));
    if (!fix_history) error->all(FLERR, "gran/spherocyl/history: could not create FixNeighHistory");
    fix_history->pair = this;   // CRITICAL: FixNeighHistory uses this pointer
  }

  // FixFreeze support (copied from parent)
  auto fixlist = modify->get_fix_by_style("^freeze");
  if (fixlist.size() == 0)
    freeze_group_bit = 0;
  else if (fixlist.size() > 1)
    error->all(FLERR, "Only one fix freeze command at a time allowed");
  else
    freeze_group_bit = fixlist.front()->groupbit;

  // FixRigid support (copied from parent)
  fix_rigid = nullptr;
  for (const auto &ifix : modify->get_fix_list()) {
    if (ifix->rigid_flag) {
      if (fix_rigid)
        error->all(FLERR, "Only one fix rigid command at a time allowed");
      else
        fix_rigid = ifix;
    }
  }

  // sanity: the fix must exist and must be wired to this pair
  if (history) {
    if (!fix_history)
      error->all(FLERR, "gran/spherocyl/history: FixNeighHistory is NULL after init_style()");
    fix_history->pair = this;   // keep this explicit for safety
 }
}

double PairGranSpherocylHistory::init_one(int i, int j)
{
  if (!allocated) allocate_sc();

  // Conservative cutoff using triangle inequality:
  // min distance between segments >= |rij| - (Hi+Hj)
  // contact possible if min distance <= (Ri+Rj)
  // => |rij| <= (Hi+Hj) + (Ri+Rj)
  double cutoff = (H[i] + H[j]) + (R[i] + R[j]);
  return cutoff;
}

inline void PairGranSpherocylHistory::axis_from_quat(const double *q, double *u) const
{
  // Convert quaternion to rotation matrix and take body z-axis as rod axis.
  double a[3][3];
  MathExtra::quat_to_mat(q, a);
  u[0] = a[0][2];
  u[1] = a[1][2];
  u[2] = a[2][2];

  // normalize defensively
  double inv = 1.0 / std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + 1e-30);
  u[0] *= inv; u[1] *= inv; u[2] *= inv;
}

inline void PairGranSpherocylHistory::closest_approach(const double *xi, const double *xj,
                                                      const double *ui, const double *uj,
                                                      double Hi, double Hj,
                                                      double *del, double *rhoi, double *rhoj,
                                                      double &rsq) const
{
  // This is the same logic as your MFIX cfrelvel kernel (Vega-Lago-style).
  // xi,xj = centers
  // ui,uj = unit axes
  // segment parameters lambda in [-Hi,Hi], mu in [-Hj,Hj]

  double rij[3] = { xj[0]-xi[0], xj[1]-xi[1], xj[2]-xi[2] };
  double rijsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2];

  double uidot = ui[0]*rij[0] + ui[1]*rij[1] + ui[2]*rij[2];
  double ujdot = uj[0]*rij[0] + uj[1]*rij[1] + uj[2]*rij[2];
  double udot  = ui[0]*uj[0]  + ui[1]*uj[1]  + ui[2]*uj[2];
  double denom = 1.0 - udot*udot;

  const double SMALL = 1e-14;
  double lambda, mu;

  if (denom < SMALL) {
    // nearly parallel
    double tmp = std::fabs(uidot) - (Hi + Hj);
    double dpar = std::max(0.0, tmp*tmp);

    // fallback for closest points
    if (uidot != 0.0) {
      lambda = std::copysign(Hi, uidot);
      mu = lambda*udot - ujdot;
      if (std::fabs(mu) > Hj) mu = std::copysign(Hj, mu);
    } else {
      lambda = 0.0; mu = 0.0;
    }

    // compute points, then rsq from actual points (more robust than shortcut)
  } else {
    double oden = 1.0/denom;
    lambda = (uidot - udot*ujdot)*oden;
    mu     = (-ujdot + udot*uidot)*oden;

    double vi = std::fabs(lambda) - Hi;
    double vj = std::fabs(mu)     - Hj;

    if (vi > 0.0 || vj > 0.0) {
      if (vi > vj) {
        lambda = std::copysign(Hi, lambda);
        mu = lambda*udot - ujdot;
        if (std::fabs(mu) > Hj) mu = std::copysign(Hj, mu);
      } else {
        mu = std::copysign(Hj, mu);
        lambda = mu*udot + uidot;
        if (std::fabs(lambda) > Hi) lambda = std::copysign(Hi, lambda);
      }
    }
  }

  // lever arms to closest points
  rhoi[0] = lambda*ui[0]; rhoi[1] = lambda*ui[1]; rhoi[2] = lambda*ui[2];
  rhoj[0] = mu*uj[0];     rhoj[1] = mu*uj[1];     rhoj[2] = mu*uj[2];

  // contact points
  double ci[3] = { xi[0] + rhoi[0], xi[1] + rhoi[1], xi[2] + rhoi[2] };
  double cj[3] = { xj[0] + rhoj[0], xj[1] + rhoj[1], xj[2] + rhoj[2] };

  // del = Ci - Cj (vector from j-contact to i-contact)
  del[0] = ci[0] - cj[0];
  del[1] = ci[1] - cj[1];
  del[2] = ci[2] - cj[2];

  rsq = del[0]*del[0] + del[1]*del[1] + del[2]*del[2];
}

void PairGranSpherocylHistory::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  int i, j, ii, jj, inum, jnum;
  double xtmp, ytmp, ztmp;
  double delx, dely, delz, fx, fy, fz;
  double ri, rj, radsum, rsq, r, rinv, rsqinv;
  double vr1, vr2, vr3, vnnr, vn1, vn2, vn3, vt1, vt2, vt3;
  double vtr1, vtr2, vtr3, vrel;
  double mi, mj, meff, damp, ccel;
  double fn, fs, fs1, fs2, fs3;
  double shrmag, rsht;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int *touch, **firsttouch;
  double *shear, *allshear, **firstshear;

  int shearupdate = 1;
  if (update->setupflag) shearupdate = 0;

  // rigid masses update (same as parent)
  if (fix_rigid && neighbor->ago == 0) {
    int tmp;
    int *body = (int *) fix_rigid->extract("body", tmp);
    auto *mass_body = (double *) fix_rigid->extract("masstotal", tmp);
    if (atom->nmax > nmax) {
      memory->destroy(mass_rigid);
      nmax = atom->nmax;
      memory->create(mass_rigid, nmax, "pair:mass_rigid");
    }
    int nlocal = atom->nlocal;
    for (i = 0; i < nlocal; i++)
      if (body[i] >= 0)
        mass_rigid[i] = mass_body[body[i]];
      else
        mass_rigid[i] = 0.0;
    comm->forward_comm(this);
  }

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **torque = atom->torque;

  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // --- Robust ellipsoid orientation access (do NOT use atom->quat) ---
  int *ellipsoid = atom->ellipsoid;
  if (ellipsoid == nullptr)
    error->all(FLERR, "gran/spherocyl/history requires atom_style ellipsoid (missing atom->ellipsoid)");

  // The ellipsoid data lives in AtomVecEllipsoid bonus storage.
  auto *avec_ellip = dynamic_cast<AtomVecEllipsoid *>(atom->avec);
  if (!avec_ellip)
    error->all(FLERR, "gran/spherocyl/history requires atom_style ellipsoid (AtomVecEllipsoid not active)");

  AtomVecEllipsoid::Bonus *bonus = avec_ellip->bonus;
  if (bonus == nullptr)
    error->all(FLERR, "gran/spherocyl/history: ellipsoid bonus storage is NULL");

  double **angmom = atom->angmom;
  if (angmom == nullptr)
    error->all(FLERR, "gran/spherocyl/history: atom->angmom is NULL. This requires ASPHERE rotational DOFs (fix nve/asphere).");

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  firsttouch = fix_history->firstflag;
  firstshear = fix_history->firstvalue;

  // loop over neighbors
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    int itype = type[i];
    ri = R[itype];

    touch = firsttouch[i];
    allshear = firstshear[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // rod axis of i (from ellipsoid bonus quaternion)
    double ui[3];
    if (ellipsoid[i] < 0)
      error->one(FLERR, i, "gran/spherocyl/history: atom i has no ellipsoid data (ellipsoid[i] < 0)");
    axis_from_quat(bonus[ellipsoid[i]].quat, ui);

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj] & NEIGHMASK;
      int jtype = type[j];

      if (!setflag[itype][jtype]) continue;

      rj = R[jtype];
      radsum = ri + rj;

      // rod axis of j (from ellipsoid bonus quaternion)
      double uj[3];
      if (ellipsoid[j] < 0)
        error->one(FLERR, j, "gran/spherocyl/history: atom j has no ellipsoid data (ellipsoid[j] < 0)");
      axis_from_quat(bonus[ellipsoid[j]].quat, uj);


      // closest approach between centerline segments
      double del[3], rhoi[3], rhoj[3];
      closest_approach(x[i], x[j], ui, uj, H[itype], H[jtype], del, rhoi, rhoj, rsq);

      // if not in contact: reset history (exactly like parent)
      if (rsq >= radsum * radsum) {
        touch[jj] = 0;
        shear = &allshear[3 * jj];
        shear[0] = 0.0;
        shear[1] = 0.0;
        shear[2] = 0.0;
        continue;
      }

      // contact normal direction uses del = Ci - Cj (from j-contact to i-contact)
      delx = del[0];
      dely = del[1];
      delz = del[2];

      r = std::sqrt(rsq);
      rinv = 1.0 / (r + 1e-30);
      rsqinv = 1.0 / (rsq + 1e-30);

      // relative velocity at contact point:
      // vc = (vi + wi x rhoi) - (vj + wj x rhoj)
      //
      // For ellipsoids/asphere, angular velocity is derived from angular momentum.
      // Use the same conversion LAMMPS uses internally: MathExtra::mq_to_omega().
      //
      // principal moments for an ellipsoid (semi-axes = shape[]) are:
      // I0 = (1/5) m (b^2 + c^2), I1 = (1/5) m (a^2 + c^2), I2 = (1/5) m (a^2 + b^2)
      // where shape[] = [a,b,c].  LAMMPS uses prefactor INERTIA=0.2 = 1/5.

      constexpr double INERTIA = 0.2;

      double omegai[3], omegaj[3];
      double inertia_i[3], inertia_j[3];

      // --- i: quat + shape from ellipsoid bonus ---
      double *shape_i = bonus[ellipsoid[i]].shape;
      double *quat_i  = bonus[ellipsoid[i]].quat;

      // mass for rotational inertia (use per-atom rmass if present, else type mass)
      double mi_rot = (atom->rmass_flag) ? rmass[i] : atom->mass[type[i]];

      // principal inertias in BODY frame
      inertia_i[0] = INERTIA * mi_rot * (shape_i[1]*shape_i[1] + shape_i[2]*shape_i[2]);
      inertia_i[1] = INERTIA * mi_rot * (shape_i[0]*shape_i[0] + shape_i[2]*shape_i[2]);
      inertia_i[2] = INERTIA * mi_rot * (shape_i[0]*shape_i[0] + shape_i[1]*shape_i[1]);

      // omega in SPACE frame from (angmom, quat, inertia)
      MathExtra::mq_to_omega(angmom[i], quat_i, inertia_i, omegai);

      // --- j: quat + shape from ellipsoid bonus ---
      double *shape_j = bonus[ellipsoid[j]].shape;
      double *quat_j  = bonus[ellipsoid[j]].quat;

      double mj_rot = (atom->rmass_flag) ? rmass[j] : atom->mass[type[j]];

      inertia_j[0] = INERTIA * mj_rot * (shape_j[1]*shape_j[1] + shape_j[2]*shape_j[2]);
      inertia_j[1] = INERTIA * mj_rot * (shape_j[0]*shape_j[0] + shape_j[2]*shape_j[2]);
      inertia_j[2] = INERTIA * mj_rot * (shape_j[0]*shape_j[0] + shape_j[1]*shape_j[1]);

      MathExtra::mq_to_omega(angmom[j], quat_j, inertia_j, omegaj);

      // now omega x rho
      double wxc_i[3] = {
        omegai[1]*rhoi[2] - omegai[2]*rhoi[1],
        omegai[2]*rhoi[0] - omegai[0]*rhoi[2],
        omegai[0]*rhoi[1] - omegai[1]*rhoi[0]
      };
      double wxc_j[3] = {
        omegaj[1]*rhoj[2] - omegaj[2]*rhoj[1],
        omegaj[2]*rhoj[0] - omegaj[0]*rhoj[2],
        omegaj[0]*rhoj[1] - omegaj[1]*rhoj[0]
      };

      vr1 = (v[i][0] + wxc_i[0]) - (v[j][0] + wxc_j[0]);
      vr2 = (v[i][1] + wxc_i[1]) - (v[j][1] + wxc_j[1]);
      vr3 = (v[i][2] + wxc_i[2]) - (v[j][2] + wxc_j[2]);

      // normal component (same algebra as parent)
      vnnr = vr1 * delx + vr2 * dely + vr3 * delz;
      vn1 = delx * vnnr * rsqinv;
      vn2 = dely * vnnr * rsqinv;
      vn3 = delz * vnnr * rsqinv;

      // tangential component at contact
      vt1 = vr1 - vn1;
      vt2 = vr2 - vn2;
      vt3 = vr3 - vn3;

      // effective mass logic (copied from parent)
      if (atom->rmass_flag) {
  	mi = rmass[i];
  	mj = rmass[j];
      } else {
  	mi = atom->mass[type[i]];
  	mj = atom->mass[type[j]];
      }

      if (fix_rigid) {
        if (mass_rigid[i] > 0.0) mi = mass_rigid[i];
        if (mass_rigid[j] > 0.0) mj = mass_rigid[j];
      }

      if (mask[i] & freeze_group_bit) mi = 1.0e20;
      if (mask[j] & freeze_group_bit) mj = 1.0e20;

      meff = mi * mj / (mi + mj);

      // normal damping (same form as parent: gamman * meff * vn)
      damp = meff * gamman * vnnr * rsqinv;

      // Hooke normal force coefficient (same as parent)
      ccel = kn * (radsum - r) * rinv - damp;
      if (limit_damping && (ccel < 0.0)) ccel = 0.0;

      // tangential relative velocity used for history (same as parent, but now vt already includes rotation)
      vtr1 = vt1;
      vtr2 = vt2;
      vtr3 = vt3;

      vrel = std::sqrt(vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3);

      // shear history effects (identical to parent)
      touch[jj] = 1;
      shear = &allshear[3 * jj];

      if (shearupdate) {
        shear[0] += vtr1 * dt;
        shear[1] += vtr2 * dt;
        shear[2] += vtr3 * dt;
      }

      shrmag = std::sqrt(shear[0]*shear[0] + shear[1]*shear[1] + shear[2]*shear[2]);

      if (shearupdate) {
        // rotate shear displacement to remain tangential to current normal
        rsht = shear[0]*delx + shear[1]*dely + shear[2]*delz;
        rsht *= rsqinv;
        shear[0] -= rsht * delx;
        shear[1] -= rsht * dely;
        shear[2] -= rsht * delz;
      }

      // tangential forces = shear spring + tangential damping (identical)
      fs1 = -(kt * shear[0] + meff * gammat * vtr1);
      fs2 = -(kt * shear[1] + meff * gammat * vtr2);
      fs3 = -(kt * shear[2] + meff * gammat * vtr3);

      // Coulomb limit (identical)
      fs = std::sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
      fn = xmu * std::fabs(ccel * r);  // same pattern as parent (ccel*r gives |Fn|)
      if (fs > fn) {
        if (fs > 0.0) {
          double scale = fn / fs;
          fs1 *= scale;
          fs2 *= scale;
          fs3 *= scale;

          // rescale shear displacement consistently
          if (kt > 0.0) {
            shear[0] = -(fs1 + meff*gammat*vtr1) / kt;
            shear[1] = -(fs2 + meff*gammat*vtr2) / kt;
            shear[2] = -(fs3 + meff*gammat*vtr3) / kt;
          }
        }
      }

      // total force on i from j (same combination form)
      fx = delx * ccel + fs1;
      fy = dely * ccel + fs2;
      fz = delz * ccel + fs3;

      // apply to forces
      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;

      // torque about centers using lever arms: tau = rho x F
      double taui[3] = {
        rhoi[1]*fz - rhoi[2]*fy,
        rhoi[2]*fx - rhoi[0]*fz,
        rhoi[0]*fy - rhoi[1]*fx
      };
      torque[i][0] += taui[0];
      torque[i][1] += taui[1];
      torque[i][2] += taui[2];

      if (force->newton_pair || j < nlocal) {
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;

        // opposite torque on j: tau_j = rhoj x (-F) = -(rhoj x F)
        double tauj[3] = {
          -(rhoj[1]*fz - rhoj[2]*fy),
          -(rhoj[2]*fx - rhoj[0]*fz),
          -(rhoj[0]*fy - rhoj[1]*fx)
        };
        torque[j][0] += tauj[0];
        torque[j][1] += tauj[1];
        torque[j][2] += tauj[2];
      }

      if (evflag)
        ev_tally_xyz(i, j, nlocal, force->newton_pair, 0.0, 0.0, fx, fy, fz, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

