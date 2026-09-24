#include "pair_gran_spherocyl_mfix_history.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "fix_neigh_history.h"
#include "fix_spherocyl_diag.h"
#include "force.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

namespace {
inline void axis_from_quat_local(const double *q, double *u)
{
  double a[3][3];
  MathExtra::quat_to_mat(q, a);
  u[0] = a[0][2];
  u[1] = a[1][2];
  u[2] = a[2][2];

  const double inv = 1.0 / std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + 1e-30);
  u[0] *= inv;
  u[1] *= inv;
  u[2] *= inv;
}

inline void closest_approach_local(const double *xi, const double *xj, const double *ui,
                                   const double *uj, double Hi, double Hj, double *del,
                                   double *rhoi, double *rhoj, double &rsq)
{
  double rij[3] = {xj[0] - xi[0], xj[1] - xi[1], xj[2] - xi[2]};

  const double uidot = ui[0] * rij[0] + ui[1] * rij[1] + ui[2] * rij[2];
  const double ujdot = uj[0] * rij[0] + uj[1] * rij[1] + uj[2] * rij[2];
  const double udot = ui[0] * uj[0] + ui[1] * uj[1] + ui[2] * uj[2];
  const double denom = 1.0 - udot * udot;

  const double small = 1e-14;
  double lambda, mu;

  if (denom < small) {
    if (uidot != 0.0) {
      lambda = std::copysign(Hi, uidot);
      mu = lambda * udot - ujdot;
      if (std::fabs(mu) > Hj) mu = std::copysign(Hj, mu);
    } else {
      lambda = 0.0;
      mu = 0.0;
    }
  } else {
    const double oden = 1.0 / denom;
    lambda = (uidot - udot * ujdot) * oden;
    mu = (-ujdot + udot * uidot) * oden;

    const double vi = std::fabs(lambda) - Hi;
    const double vj = std::fabs(mu) - Hj;

    if (vi > 0.0 || vj > 0.0) {
      if (vi > vj) {
        lambda = std::copysign(Hi, lambda);
        mu = lambda * udot - ujdot;
        if (std::fabs(mu) > Hj) mu = std::copysign(Hj, mu);
      } else {
        mu = std::copysign(Hj, mu);
        lambda = mu * udot + uidot;
        if (std::fabs(lambda) > Hi) lambda = std::copysign(Hi, lambda);
      }
    }
  }

  rhoi[0] = lambda * ui[0];
  rhoi[1] = lambda * ui[1];
  rhoi[2] = lambda * ui[2];
  rhoj[0] = mu * uj[0];
  rhoj[1] = mu * uj[1];
  rhoj[2] = mu * uj[2];

  const double ci[3] = {xi[0] + rhoi[0], xi[1] + rhoi[1], xi[2] + rhoi[2]};
  const double cj[3] = {xj[0] + rhoj[0], xj[1] + rhoj[1], xj[2] + rhoj[2]};

  del[0] = ci[0] - cj[0];
  del[1] = ci[1] - cj[1];
  del[2] = ci[2] - cj[2];

  rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
}
}    // namespace

PairGranSpherocylMfixHistory::PairGranSpherocylMfixHistory(LAMMPS *lmp)
    : PairGranSpherocylHistory(lmp)
{
}

void PairGranSpherocylMfixHistory::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  int i, j, ii, jj, inum, jnum;
  double delx, dely, delz, fx, fy, fz;
  double ri, rj, radsum, rsq, r, rinv, rsqinv;
  double vr1, vr2, vr3, vnnr;
  double mi, mj, damp, ccel;
  double delta, delta_sqrt, delta_quarter;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int *touch, **firsttouch;
  double *shear, *allshear, **firstshear;

  events_new_local_step = 0;
  if (diag) diag->pair_begin();

  if (fix_rigid && neighbor->ago == 0) {
    int tmp;
    int *body = (int *) fix_rigid->extract("body", tmp);
    auto *mass_body = (double *) fix_rigid->extract("masstotal", tmp);
    if (atom->nmax > nmax) {
      memory->destroy(mass_rigid);
      nmax = atom->nmax;
      memory->create(mass_rigid, nmax, "pair:mass_rigid");
    }
    const int nlocal = atom->nlocal;
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
  const int nlocal = atom->nlocal;

  int *ellipsoid = atom->ellipsoid;
  if (ellipsoid == nullptr)
    error->all(FLERR,
               "gran/spherocyl/mfix/history requires atom_style ellipsoid "
               "(missing atom->ellipsoid)");

  auto *avec_ellip = dynamic_cast<AtomVecEllipsoid *>(atom->avec);
  if (!avec_ellip)
    error->all(FLERR,
               "gran/spherocyl/mfix/history requires atom_style ellipsoid "
               "(AtomVecEllipsoid not active)");

  AtomVecEllipsoid::Bonus *bonus = avec_ellip->bonus;
  if (bonus == nullptr)
    error->all(FLERR, "gran/spherocyl/mfix/history: ellipsoid bonus storage is NULL");

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  firsttouch = fix_history->firstflag;
  firstshear = fix_history->firstvalue;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    const int itype = type[i];
    ri = R[itype];

    touch = firsttouch[i];
    allshear = firstshear[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    double ui[3];
    if (ellipsoid[i] < 0)
      error->one(FLERR, i,
                 "gran/spherocyl/mfix/history: atom i has no ellipsoid data "
                 "(ellipsoid[i] < 0)");
    axis_from_quat_local(bonus[ellipsoid[i]].quat, ui);

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj] & NEIGHMASK;
      const int jtype = type[j];

      if (!setflag[itype][jtype]) continue;

      rj = R[jtype];
      radsum = ri + rj;

      double uj[3];
      if (ellipsoid[j] < 0)
        error->one(FLERR, j,
                   "gran/spherocyl/mfix/history: atom j has no ellipsoid data "
                   "(ellipsoid[j] < 0)");
      axis_from_quat_local(bonus[ellipsoid[j]].quat, uj);

      double del[3], rhoi[3], rhoj[3];
      closest_approach_local(x[i], x[j], ui, uj, H[itype], H[jtype], del, rhoi, rhoj, rsq);

      double *hist = &allshear[size_history * jj];
      shear = hist;
      if (rsq >= radsum * radsum) {
        if (touch[jj] && diag) diag_release(i, j, nlocal, hist, rhoi, rhoj);
        touch[jj] = 0;
        memset(hist, 0, size_history * sizeof(double));
        continue;
      }

      delx = del[0];
      dely = del[1];
      delz = del[2];

      r = std::sqrt(rsq);
      rinv = 1.0 / (r + 1e-30);
      rsqinv = 1.0 / (rsq + 1e-30);
      delta = radsum - r;
      delta_sqrt = std::sqrt(delta);
      delta_quarter = std::sqrt(delta_sqrt);

      vr1 = v[i][0] - v[j][0];
      vr2 = v[i][1] - v[j][1];
      vr3 = v[i][2] - v[j][2];
      vnnr = vr1 * delx + vr2 * dely + vr3 * delz;

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

      damp = gamman * delta_quarter * vnnr * rsqinv;
      ccel = kn * delta_sqrt * delta * rinv - damp;
      if (limit_damping && (ccel < 0.0)) ccel = 0.0;

      const int newcontact = (touch[jj] == 0);
      if (newcontact) {
        const bool owns_contact =
          force->newton_pair || j < nlocal || tag[i] < tag[j];
        if (owns_contact) events_new_local_step++;
      }
      touch[jj] = 1;
      shear[0] = 0.0;
      shear[1] = 0.0;
      shear[2] = 0.0;

      fx = delx * ccel;
      fy = dely * ccel;
      fz = delz * ccel;

      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;

      double taui[3] = {
        rhoi[1] * fz - rhoi[2] * fy,
        rhoi[2] * fx - rhoi[0] * fz,
        rhoi[0] * fy - rhoi[1] * fx
      };
      torque[i][0] += taui[0];
      torque[i][1] += taui[1];
      torque[i][2] += taui[2];

      if (force->newton_pair || j < nlocal) {
        f[j][0] -= fx;
        f[j][1] -= fy;
        f[j][2] -= fz;

        double tauj[3] = {
          -(rhoj[1] * fz - rhoj[2] * fy),
          -(rhoj[2] * fx - rhoj[0] * fz),
          -(rhoj[0] * fy - rhoj[1] * fx)
        };
        torque[j][0] += tauj[0];
        torque[j][1] += tauj[1];
        torque[j][2] += tauj[2];
      }

      if (diag) {
        const double F[3] = {fx, fy, fz};
        diag_contact(i, j, nlocal, newcontact, hist, del, r, radsum, rhoi, rhoj, ui, uj, F,
                     kn * delta_sqrt * delta, 0.4 * kn * delta * delta * delta_sqrt);
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
