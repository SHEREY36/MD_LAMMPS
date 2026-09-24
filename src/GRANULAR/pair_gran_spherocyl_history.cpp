#include "pair_gran_spherocyl_history.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "fix_dummy.h"
#include "fix_neigh_history.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "atom_vec_ellipsoid.h"
#include "fix_spherocyl_diag.h"

#include <cmath>
#include <cstdio>
#include <cstring>

using namespace LAMMPS_NS;
using MathConst::MY_PI;

// Principal moments for a 3D spherocylinder (capsule):
// diameter = d = 2R, cylindrical section length = lcyl = 2H.
// Axis of symmetry is body-z, so Ixx=Iyy=I_perp and Izz=I_parallel.
void PairGranSpherocylHistory::spherocyl_inertia(double mass, double radius, double half_cyl,
                                                 double *inertia)
{
  const double d = 2.0 * radius;
  const double lcyl = 2.0 * half_cyl;

  const double d2 = d * d;
  const double d3 = d2 * d;
  const double d4 = d2 * d2;
  const double d5 = d4 * d;
  const double lcyl2 = lcyl * lcyl;
  const double lcyl3 = lcyl2 * lcyl;

  // rho from particle mass and spherocylinder volume
  const double vol = MY_PI * d3 / 6.0 + MY_PI * d2 * lcyl / 4.0;
  const double rho = mass / (vol + 1e-30);

  const double iperp =
      (MY_PI / 48.0) * rho * d2 * lcyl3 +
      (3.0 * MY_PI / 64.0) * rho * d4 * lcyl +
      (MY_PI / 60.0) * rho * d5 +
      (MY_PI / 24.0) * rho * d3 * lcyl2;

  const double ipar =
      (MY_PI / 32.0) * rho * d4 * lcyl +
      (MY_PI / 60.0) * rho * d5;

  inertia[0] = iperp;
  inertia[1] = iperp;
  inertia[2] = ipar;
}

PairGranSpherocylHistory::PairGranSpherocylHistory(LAMMPS *lmp)
    : PairGranHookeHistory(lmp)
{
  R = nullptr;
  H = nullptr;
  events_new_local_step = 0;
  events_new_global_step = 0;
  events_cum_local = 0;
  events_cum_global = 0;
  events_cum_global_prev_log = 0;
  events_last_logged_step = -1;
  events_log_every = 1000;
  events_fp = nullptr;
  diag = nullptr;
  // shear displacement (3) + per-contact collision record for fix spherocyl/diag
  size_history = NSHEAR + NDIAG;
  // The neighbor list is NOT a size (REQ_SIZE) list (atom_style ellipsoid has no
  // radius), so NPair never sets the "touching" history bit.  Without this flag
  // FixNeighHistory::post_neighbor() drops the history (touch flag, shear
  // displacement, collision record) of every contact that is in progress when
  // the neighbor list is rebuilt.  With beyond_contact = 1 the history of every
  // pair found in the old partner lists (= pairs that were touching) is kept,
  // exactly as pair_style granular does when it does not request REQ_SIZE.
  beyond_contact = 1;
}

PairGranSpherocylHistory::~PairGranSpherocylHistory()
{
  if (events_fp) {
    fclose(events_fp);
    events_fp = nullptr;
  }
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

  // Do NOT use REQ_SIZE: atom_style ellipsoid has no atom->radius array (it is NULL).
  // REQ_SIZE makes NPairBin access atom->radius[i], which crashes with address-not-mapped.
  // Use fixed geometric cutoff from init_one() instead.
  if (history)
    neighbor->add_request(this, NeighConst::REQ_HISTORY);
  else
    neighbor->add_request(this);

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

  const bool fresh_events_state =
      (events_last_logged_step < 0 && events_cum_local == 0 && events_cum_global == 0 &&
       events_cum_global_prev_log == 0);

  // Open collision-event log on rank 0 once per pair object.
  // init_style() can be called again during operations like write_restart;
  // reopening with "w" there would silently truncate the accumulated log.
  if (comm->me == 0 && events_fp == nullptr) {
    events_fp = fopen("collision_events.dat", "w");
    if (!events_fp)
      error->one(FLERR, "gran/spherocyl/history: could not open collision_events.dat");
    fprintf(events_fp, "# step new_events_since_last_log global_cumulative_events\n");
    fflush(events_fp);
  }

  if (fresh_events_state) {
    events_new_local_step = 0;
    events_new_global_step = 0;
    events_cum_local = 0;
    events_cum_global = 0;
    events_cum_global_prev_log = 0;
    events_last_logged_step = -1;
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

void *PairGranSpherocylHistory::extract(const char *str, int &dim)
{
  dim = 1;
  if (strcmp(str, "spherocyl_R") == 0) return (void *) R;
  if (strcmp(str, "spherocyl_H") == 0) return (void *) H;
  return nullptr;
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
  // Mirror OVERLAP_PP geometry logic used in your Fortran collision model.
  // xi,xj = centers
  // ui,uj = unit axes
  // segment parameters lambda in [-Hi,Hi], mu in [-Hj,Hj]

  double rij[3] = { xj[0]-xi[0], xj[1]-xi[1], xj[2]-xi[2] };

  double uidot = ui[0]*rij[0] + ui[1]*rij[1] + ui[2]*rij[2];
  double ujdot = uj[0]*rij[0] + uj[1]*rij[1] + uj[2]*rij[2];
  double udot  = ui[0]*uj[0]  + ui[1]*uj[1]  + ui[2]*uj[2];
  double denom = 1.0 - udot*udot;

  const double SMALL = 1e-14;
  double lambda, mu;

  if (denom < SMALL) {
    // nearly parallel
    // Keep the same branch structure and clamping behavior as OVERLAP_PP.
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

  // Lever arms to closest points on centerlines
  rhoi[0] = lambda*ui[0]; rhoi[1] = lambda*ui[1]; rhoi[2] = lambda*ui[2];
  rhoj[0] = mu*uj[0];     rhoj[1] = mu*uj[1];     rhoj[2] = mu*uj[2];

  // Closest points
  double ci[3] = { xi[0] + rhoi[0], xi[1] + rhoi[1], xi[2] + rhoi[2] };
  double cj[3] = { xj[0] + rhoj[0], xj[1] + rhoj[1], xj[2] + rhoj[2] };

  // Keep PairGran sign convention: del = Ci - Cj (j -> i)
  del[0] = ci[0] - cj[0];
  del[1] = ci[1] - cj[1];
  del[2] = ci[2] - cj[2];

  rsq = del[0]*del[0] + del[1]*del[1] + del[2]*del[2];
}

void PairGranSpherocylHistory::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  int i, j, ii, jj, inum, jnum;
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

  events_new_local_step = 0;
  if (diag) diag->pair_begin();

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
  tagint *tag = atom->tag;
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
    error->all(FLERR, "gran/spherocyl/history: atom->angmom is NULL. This requires rotational DOFs (fix nve/spherocyl or fix nve/asphere).");

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  firsttouch = fix_history->firstflag;
  firstshear = fix_history->firstvalue;

  // loop over neighbors
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

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

      double *hist = &allshear[size_history * jj];
      shear = hist;

      // if not in contact: close the collision record, then reset history
      if (rsq >= radsum * radsum) {
        if (touch[jj] && diag) diag_release(i, j, nlocal, hist, rhoi, rhoj);
        touch[jj] = 0;
        memset(hist, 0, size_history * sizeof(double));
        continue;
      }

      // contact normal direction uses del = Ci - Cj (from j-contact to i-contact)
      delx = del[0];
      dely = del[1];
      delz = del[2];

      r = std::sqrt(rsq);
      rinv = 1.0 / (r + 1e-30);
      rsqinv = 1.0 / (rsq + 1e-30);

      // translational-only relative velocity for the normal channel
      // (omega x rho omitted by design: gamma_n is defined for translational v_rel)
      vr1 = v[i][0] - v[j][0];
      vr2 = v[i][1] - v[j][1];
      vr3 = v[i][2] - v[j][2];

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

      vtr1 = vt1;
      vtr2 = vt2;
      vtr3 = vt3;

      vrel = std::sqrt(vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3);

      // Count each new contact exactly once globally.
      // With half neighbor lists and newton off, local-local pairs are owned
      // by the stored i/j ordering (j < nlocal here), while local-ghost pairs
      // appear on both ranks and need a tag-based ownership filter.
      const int newcontact = (touch[jj] == 0);
      if (newcontact) {
        const bool owns_contact =
          force->newton_pair || j < nlocal || tag[i] < tag[j];
        if (owns_contact) events_new_local_step++;
      }
      touch[jj] = 1;

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

      if (diag) {
        const double F[3] = {fx, fy, fz};
        const double delta = radsum - r;
        diag_contact(i, j, nlocal, newcontact, hist, del, r, radsum, rhoi, rhoj, ui, uj, F,
                     kn * delta, 0.5 * kn * delta * delta);
      }

      // Virial with the centre-of-mass branch vector x_i - x_j (NOT the
      // contact-point separation del): the shear work and the momentum flux
      // across a plane are carried between particle centres.
      if (evflag)
        ev_tally_xyz(i, j, nlocal, force->newton_pair, 0.0, 0.0, fx, fy, fz,
                     x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]);
    }
  }

  events_cum_local += events_new_local_step;
  maybe_log_collision_events();

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   angular velocity (space frame) and rotational kinetic energy of atom i
------------------------------------------------------------------------- */

void PairGranSpherocylHistory::omega_erot(int i, double *omega, double &erot)
{
  auto *avec = dynamic_cast<AtomVecEllipsoid *>(atom->avec);
  const int itype = atom->type[i];
  const double mass = atom->rmass_flag ? atom->rmass[i] : atom->mass[itype];
  double inertia[3];
  spherocyl_inertia(mass, R[itype], H[itype], inertia);
  MathExtra::mq_to_omega(atom->angmom[i], avec->bonus[atom->ellipsoid[i]].quat, inertia, omega);
  const double *L = atom->angmom[i];
  erot = 0.5 * (L[0] * omega[0] + L[1] * omega[1] + L[2] * omega[2]);
}

/* ----------------------------------------------------------------------
   fix spherocyl/diag: per-step bookkeeping of a touching pair
   Ownership (one rank books each pair): newton on, or j local, or
   tag[i] < tag[j] for local-ghost pairs (these appear on two ranks).
------------------------------------------------------------------------- */

void PairGranSpherocylHistory::diag_contact(int i, int j, int nlocal, int newcontact,
                                            double *hist, const double *del, double r,
                                            double radsum, const double *rhoi,
                                            const double *rhoj, const double *ui,
                                            const double *uj, const double *F, double fel,
                                            double uel)
{
  FixSpherocylDiag *dg = diag;
  double *h = hist + NSHEAR;
  tagint *tag = atom->tag;
  double **x = atom->x;
  double **v = atom->v;
  double **pa = dg->pa;
  const bool owner = force->newton_pair || j < nlocal || tag[i] < tag[j];
  const bool active = !update->setupflag;
  const double dt = update->dt;
  const double tnow = update->atime + (update->ntimestep - update->atimestep) * dt;

  const double rinv = 1.0 / (r + 1e-300);
  const double nhat[3] = {del[0] * rinv, del[1] * rinv, del[2] * rinv};

  double wi[3], wj[3], eroti, erotj;
  omega_erot(i, wi, eroti);
  omega_erot(j, wj, erotj);
  const double vc[3] = {
      v[i][0] + wi[1] * rhoi[2] - wi[2] * rhoi[1] - (v[j][0] + wj[1] * rhoj[2] - wj[2] * rhoj[1]),
      v[i][1] + wi[2] * rhoi[0] - wi[0] * rhoi[2] - (v[j][1] + wj[2] * rhoj[0] - wj[0] * rhoj[2]),
      v[i][2] + wi[0] * rhoi[1] - wi[1] * rhoi[0] - (v[j][2] + wj[0] * rhoj[1] - wj[1] * rhoj[0])};

  // coordination: current-step counts for local atoms, previous-step counts
  // (forward communicated) for the "other contacts" test
  pa[i][FixSpherocylDiag::PA_ZCUR] += 1.0;
  if (j < nlocal) pa[j][FixSpherocylDiag::PA_ZCUR] += 1.0;
  double zo = MAX(pa[i][FixSpherocylDiag::PA_ZPREV], pa[j][FixSpherocylDiag::PA_ZPREV]) -
      (newcontact ? 0.0 : 1.0);
  if (zo < 0.0) zo = 0.0;

  if (newcontact || h[D_MARK] == 0.0) {
    for (int k = 0; k < NDIAG; k++) h[k] = 0.0;
    h[D_MARK] = 1.0;
    // a contact whose history was lost (e.g. pair re-created mid contact) is
    // tracked but not reported as a collision
    h[D_T0] = newcontact ? tnow : -1.0e300;
    for (int k = 0; k < 3; k++) {
      h[D_G0 + k] = v[i][k] - v[j][k];
      h[D_N0 + k] = nhat[k];
    }
    h[D_GC0] = vc[0] * nhat[0] + vc[1] * nhat[1] + vc[2] * nhat[2];
    for (int k = 0; k < 3; k++) h[D_VC0 + k] = vc[k];
    h[D_EROT0] = eroti + erotj;
    const int itype = atom->type[i], jtype = atom->type[j];
    const double li = H[itype] > 0.0 ? sqrt(rhoi[0]*rhoi[0] + rhoi[1]*rhoi[1] + rhoi[2]*rhoi[2]) / H[itype] : 0.0;
    const double lj = H[jtype] > 0.0 ? sqrt(rhoj[0]*rhoj[0] + rhoj[1]*rhoj[1] + rhoj[2]*rhoj[2]) / H[jtype] : 0.0;
    h[D_LMIN] = MIN(li, lj);
    h[D_LMAX] = MAX(li, lj);
    h[D_UU] = fabs(ui[0] * uj[0] + ui[1] * uj[1] + ui[2] * uj[2]);
    h[D_SAME] = (pa[i][FixSpherocylDiag::PA_LASTP] == (double) tag[j]) ? 1.0 : 0.0;
    if (newcontact) {
      dg->contact_start_atoms(i, j, nlocal, tnow);
      if (owner) dg->acc[FixSpherocylDiag::A_NSTART] += 1.0;
    }
  }

  const double s = h[D_MARK];    // true symmetric value = stored * s
  const double dd = (radsum - r) / radsum;
  if (dd > s * h[D_DMAX]) h[D_DMAX] = s * dd;
  if (zo > s * h[D_ZOTH]) h[D_ZOTH] = s * zo;
  if (owner) dg->uel_step += uel;

  if (!active) return;

  const double rij[3] = {x[i][0] - x[j][0], x[i][1] - x[j][1], x[i][2] - x[j][2]};
  const double Fnc[3] = {F[0] - fel * nhat[0], F[1] - fel * nhat[1], F[2] - fel * nhat[2]};
  // per-contact non-elastic work (half-step velocities; O(dt) accurate, used
  // only for the per-collision record and the energy-gain indicator)
  const double pnc = (Fnc[0] * vc[0] + Fnc[1] * vc[1] + Fnc[2] * vc[2]) * dt;

  // per-atom non-elastic force and torque for the exact energy bookkeeping
  // (power taken at end_of_step with full-step peculiar velocities)
  {
    double *fi = &pa[i][FixSpherocylDiag::PA_FNC], *ti = &pa[i][FixSpherocylDiag::PA_TNC];
    fi[0] += Fnc[0];
    fi[1] += Fnc[1];
    fi[2] += Fnc[2];
    ti[0] += rhoi[1] * Fnc[2] - rhoi[2] * Fnc[1];
    ti[1] += rhoi[2] * Fnc[0] - rhoi[0] * Fnc[2];
    ti[2] += rhoi[0] * Fnc[1] - rhoi[1] * Fnc[0];
    if (j < nlocal) {
      double *fj = &pa[j][FixSpherocylDiag::PA_FNC], *tj = &pa[j][FixSpherocylDiag::PA_TNC];
      fj[0] -= Fnc[0];
      fj[1] -= Fnc[1];
      fj[2] -= Fnc[2];
      tj[0] -= rhoj[1] * Fnc[2] - rhoj[2] * Fnc[1];
      tj[1] -= rhoj[2] * Fnc[0] - rhoj[0] * Fnc[2];
      tj[2] -= rhoj[0] * Fnc[1] - rhoj[1] * Fnc[0];
    }
  }
  h[D_WNC] += s * pnc;
  if (pnc > 0.0) h[D_WPOS] += s * pnc;
  for (int a = 0; a < 3; a++) {
    h[D_J + a] += F[a] * dt;
    for (int b = 0; b < 3; b++) h[D_RF + 3 * a + b] += s * rij[a] * F[b] * dt;
  }

  if (owner) {
    double *acc = dg->acc;
    const double *G = dg->G;    // Voigt: xx yy zz yz xz xy
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) acc[FixSpherocylDiag::A_C + 3 * a + b] += rij[a] * F[b] * dt;
    // collisional shear work: -sum_ab G_ab F_a r_b
    acc[FixSpherocylDiag::A_WSHC] -=
        (G[0] * F[0] * rij[0] + G[1] * F[1] * rij[1] + G[2] * F[2] * rij[2] +
         G[5] * F[0] * rij[1] + G[4] * F[0] * rij[2] + G[3] * F[1] * rij[2]) * dt;
    // streaming part of the non-elastic work: F_nc . (G r_ij)
    acc[FixSpherocylDiag::A_WNCG] +=
        (Fnc[0] * (G[0] * rij[0] + G[5] * rij[1] + G[4] * rij[2]) +
         Fnc[1] * (G[1] * rij[1] + G[3] * rij[2]) + Fnc[2] * G[2] * rij[2]) * dt;
    if (pnc > 0.0) acc[FixSpherocylDiag::A_WPOS] += pnc;
  }
}

/* ----------------------------------------------------------------------
   fix spherocyl/diag: a touching pair has just separated -> close record
   (velocities in the force stage are half-step velocities; at separation
   they already contain the full impulse of the collision)
------------------------------------------------------------------------- */

void PairGranSpherocylHistory::diag_release(int i, int j, int nlocal, double *hist,
                                            const double *rhoi, const double *rhoj)
{
  FixSpherocylDiag *dg = diag;
  double *h = hist + NSHEAR;
  const double s = h[D_MARK];
  if (s == 0.0) return;

  tagint *tag = atom->tag;
  double **v = atom->v;
  const double dt = update->dt;
  const double tnow = update->atime + (update->ntimestep - update->atimestep) * dt;

  dg->contact_end_atoms(i, j, nlocal, tnow);

  const bool owner = force->newton_pair || j < nlocal || tag[i] < tag[j];
  if (!owner) return;
  const double t0 = s * h[D_T0];
  if (t0 < -1.0e299) return;

  double wi[3], wj[3], eroti, erotj;
  omega_erot(i, wi, eroti);
  omega_erot(j, wj, erotj);
  const double g1[3] = {v[i][0] - v[j][0], v[i][1] - v[j][1], v[i][2] - v[j][2]};
  const double vc1[3] = {
      g1[0] + wi[1] * rhoi[2] - wi[2] * rhoi[1] - (wj[1] * rhoj[2] - wj[2] * rhoj[1]),
      g1[1] + wi[2] * rhoi[0] - wi[0] * rhoi[2] - (wj[2] * rhoj[0] - wj[0] * rhoj[2]),
      g1[2] + wi[0] * rhoi[1] - wi[1] * rhoi[0] - (wj[0] * rhoj[1] - wj[1] * rhoj[0])};

  const double *g0 = &h[D_G0];
  const double *n0 = &h[D_N0];
  const double *vc0 = &h[D_VC0];
  // Restitution axis = impulse direction J/|J|.  The relative-velocity change
  // of the collision is exactly parallel to J, so e = -g_post.J/g_pre.J is the
  // restitution of the equivalent instantaneous collision (the DSMC rule).
  // Projecting on the initial normal n0 instead is biased low for oblique and
  // grazing soft contacts, whose normal rotates during the contact.
  double jhat[3] = {h[D_J], h[D_J + 1], h[D_J + 2]};
  const double jmag = sqrt(jhat[0] * jhat[0] + jhat[1] * jhat[1] + jhat[2] * jhat[2]);
  for (int k = 0; k < 3; k++) jhat[k] = jmag > 0.0 ? jhat[k] / jmag : n0[k];
  const double gn_pre = g0[0] * jhat[0] + g0[1] * jhat[1] + g0[2] * jhat[2];
  const double gn_post = g1[0] * jhat[0] + g1[1] * jhat[1] + g1[2] * jhat[2];
  const double gc_pre = vc0[0] * jhat[0] + vc0[1] * jhat[1] + vc0[2] * jhat[2];
  const double gc_post = vc1[0] * jhat[0] + vc1[1] * jhat[1] + vc1[2] * jhat[2];
  const double gt_pre = sqrt(MAX(0.0, g0[0]*g0[0] + g0[1]*g0[1] + g0[2]*g0[2] - gn_pre*gn_pre));

  const int itype = atom->type[i], jtype = atom->type[j];
  const double mi = atom->rmass_flag ? atom->rmass[i] : atom->mass[itype];
  const double mj = atom->rmass_flag ? atom->rmass[j] : atom->mass[jtype];
  const double mu = mi * mj / (mi + mj);
  const double erel0 = 0.5 * mu * (g0[0]*g0[0] + g0[1]*g0[1] + g0[2]*g0[2]);
  const double erel1 = 0.5 * mu * (g1[0]*g1[0] + g1[1]*g1[1] + g1[2]*g1[2]);
  const double erot0 = s * h[D_EROT0];
  const double erot1 = eroti + erotj;
  const double epre = erel0 + erot0;
  const double dE = (erel1 + erot1) - epre;
  const double dur_t = tnow - t0;
  const double dur = dur_t / dt;
  const double dmax = s * h[D_DMAX];
  const double zo = s * h[D_ZOTH];
  const double lmin = s * h[D_LMIN], lmax = s * h[D_LMAX];
  const double uu = s * h[D_UU], same = s * h[D_SAME];

  double *acc = dg->acc;
  double *accmin = dg->accmin;
  using FD = FixSpherocylDiag;
  acc[FD::A_NEND] += 1.0;
  const bool binary = (zo < 0.5);
  if (binary) {
    acc[FD::A_NBIN] += 1.0;
    if (gn_pre < 0.0) {
      acc[FD::A_ETR] += gn_post;
      acc[FD::A_ETR2] += gn_pre;
      acc[FD::A_NETR] += 1.0;
      const double e = -gn_post / gn_pre;
      if (e < 0.0 || e > 1.0) acc[FD::A_ETROUT] += 1.0;
    }
    if (gc_pre < 0.0) {
      acc[FD::A_EC] += gc_post;
      acc[FD::A_EC2] += gc_pre;
      acc[FD::A_NEC] += 1.0;
      const double e = -gc_post / gc_pre;
      if (e < 0.0 || e > 1.0) acc[FD::A_ECOUT] += 1.0;
    }
  } else
    acc[FD::A_NMULTI] += 1.0;
  acc[FD::A_GN] += fabs(gn_pre);
  acc[FD::A_GN2] += gn_pre * gn_pre;
  acc[FD::A_DUR] += dur;
  acc[FD::A_DURT] += dur_t;
  dg->add_duration(dur);
  if (dur < dg->dur_min_thresh) acc[FD::A_NSHORT] += 1.0;
  if (dur > dg->dur_long_thresh) acc[FD::A_NLONG] += 1.0;
  accmin[FD::M_DURMIN] = MIN(accmin[FD::M_DURMIN], dur);
  accmin[FD::M_NEGDURMAX] = MIN(accmin[FD::M_NEGDURMAX], -dur);
  acc[FD::A_DMAX] += dmax;
  accmin[FD::M_NEGDMAXMAX] = MIN(accmin[FD::M_NEGDMAXMAX], -dmax);
  if (epre > 0.0) {
    if (dE > 1.0e-3 * epre) acc[FD::A_NANTI] += 1.0;
    acc[FD::A_DEFRAC] += dE / epre;
  }
  acc[FD::A_DETR] += erel1 - erel0;
  acc[FD::A_DEROT] += erot1 - erot0;
  acc[FD::A_NSAME] += same;
  if (lmax > 0.999) acc[FD::A_NTIP] += 1.0;
  acc[FD::A_UU] += uu;
  acc[FD::A_NN + 0] += n0[0] * n0[0];
  acc[FD::A_NN + 1] += n0[1] * n0[1];
  acc[FD::A_NN + 2] += n0[2] * n0[2];
  acc[FD::A_NN + 3] += n0[0] * n0[1];
  acc[FD::A_NN + 4] += n0[0] * n0[2];
  acc[FD::A_NN + 5] += n0[1] * n0[2];
  for (int k = 0; k < 9; k++) acc[FD::A_CE + k] += s * h[D_RF + k];

  if (dg->sample_pair(tag[i], tag[j])) {
    double rec[FD::NEV] = {tnow,
                           (double) tag[i],
                           (double) tag[j],
                           dur,
                           gn_pre,
                           gn_post,
                           gn_pre < 0.0 ? -gn_post / gn_pre : 0.0,
                           gc_pre,
                           gc_post,
                           gc_pre < 0.0 ? -gc_post / gc_pre : 0.0,
                           gt_pre,
                           erel0,
                           erel1,
                           erot0,
                           erot1,
                           s * h[D_WNC],
                           dmax,
                           lmin,
                           lmax,
                           uu,
                           zo,
                           same};
    dg->push_event(rec);
  }
}

void PairGranSpherocylHistory::maybe_log_collision_events()
{
  if (events_log_every <= 0) return;

  const long long step = static_cast<long long>(update->ntimestep);
  const bool at_interval = (step % events_log_every == 0);
  const bool at_run_end = (step == static_cast<long long>(update->laststep));
  if (!at_interval && !at_run_end) return;
  if (step == events_last_logged_step) return;

  MPI_Allreduce(&events_cum_local, &events_cum_global, 1, MPI_LONG_LONG, MPI_SUM, world);
  events_new_global_step = events_cum_global - events_cum_global_prev_log;

  if (comm->me == 0 && events_fp) {
    fprintf(events_fp, "%lld %lld %lld\n", step, events_new_global_step, events_cum_global);
    fflush(events_fp);
  }
  events_cum_global_prev_log = events_cum_global;
  events_last_logged_step = step;
}
