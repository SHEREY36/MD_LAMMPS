/* ----------------------------------------------------------------------
   fix_count_collisions.cpp

   Standalone collision-event counter for spherocylinder DEM.
------------------------------------------------------------------------- */

#include "fix_count_collisions.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_extra.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixCountCollisions::FixCountCollisions(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  if (narg < 4)
    error->all(FLERR,
               "Illegal fix count/collisions command.\n"
               "Usage: fix ID group-ID count/collisions log_every");

  log_every = utils::inumeric(FLERR, arg[3], false, lmp);
  if (log_every <= 0) error->all(FLERR, "fix count/collisions: log_every must be > 0");

  events_cum = 0;
  events_cum_prev = 0;
  last_logged_step = -1;
  fp = nullptr;
  counter_started = 0;
  R = nullptr;
  H = nullptr;

  nevery = 1;
}

/* ---------------------------------------------------------------------- */

FixCountCollisions::~FixCountCollisions()
{
  if (fp) {
    fclose(fp);
    fp = nullptr;
  }
}

/* ---------------------------------------------------------------------- */

int FixCountCollisions::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCountCollisions::init()
{
  if (!force->pair)
    error->all(FLERR,
               "fix count/collisions requires a pair style with "
               "spherocyl_R and spherocyl_H extractable parameters");

  int dim = 0;
  R = (double *) force->pair->extract("spherocyl_R", dim);
  H = (double *) force->pair->extract("spherocyl_H", dim);
  if (!R || !H)
    error->all(FLERR,
               "fix count/collisions could not extract spherocylinder geometry "
               "from pair style");

  if (!counter_started) {
    if (fp) {
      fclose(fp);
      fp = nullptr;
    }
    if (comm->me == 0) {
      fp = fopen("collision_count.dat", "w");
      if (!fp) error->one(FLERR, "fix count/collisions: could not open collision_count.dat");
      fprintf(fp, "# step new_events_since_last_log global_cumulative_events\n");
      fflush(fp);
    }

    active_contacts.clear();
    seen_this_step.clear();
    events_cum = 0;
    events_cum_prev = 0;
    last_logged_step = -1;
    counter_started = 1;
  }
}

/* ---------------------------------------------------------------------- */

void FixCountCollisions::end_of_step()
{
  scan_contacts();
}

/* ---------------------------------------------------------------------- */

void FixCountCollisions::scan_contacts()
{
  NeighList *list = force->pair->list;
  if (!list) return;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int nlocal = atom->nlocal;

  double **x = atom->x;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int *mask = atom->mask;

  int *ellipsoid = atom->ellipsoid;
  auto *avec = dynamic_cast<AtomVecEllipsoid *>(atom->avec);
  if (!avec) return;
  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  if (!bonus) return;

  seen_this_step.clear();
  long long new_events_this_step = 0;

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    if (ellipsoid[i] < 0) continue;

    int itype = type[i];
    double ri = R[itype];
    double Hi = H[itype];

    double ui[3];
    double *qi = bonus[ellipsoid[i]].quat;
    double ai[3][3];
    MathExtra::quat_to_mat(qi, ai);
    ui[0] = ai[0][2];
    ui[1] = ai[1][2];
    ui[2] = ai[2][2];
    double inv = 1.0 / std::sqrt(ui[0] * ui[0] + ui[1] * ui[1] + ui[2] * ui[2] + 1.0e-30);
    ui[0] *= inv;
    ui[1] *= inv;
    ui[2] *= inv;

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      if (!(mask[j] & groupbit)) continue;
      if (ellipsoid[j] < 0) continue;

      int jtype = type[j];
      double rj = R[jtype];
      double Hj = H[jtype];
      double radsum = ri + rj;

      double uj[3];
      double *qj = bonus[ellipsoid[j]].quat;
      double aj[3][3];
      MathExtra::quat_to_mat(qj, aj);
      uj[0] = aj[0][2];
      uj[1] = aj[1][2];
      uj[2] = aj[2][2];
      inv = 1.0 / std::sqrt(uj[0] * uj[0] + uj[1] * uj[1] + uj[2] * uj[2] + 1.0e-30);
      uj[0] *= inv;
      uj[1] *= inv;
      uj[2] *= inv;

      double rij[3] = {x[j][0] - x[i][0], x[j][1] - x[i][1], x[j][2] - x[i][2]};
      double uidot = ui[0] * rij[0] + ui[1] * rij[1] + ui[2] * rij[2];
      double ujdot = uj[0] * rij[0] + uj[1] * rij[1] + uj[2] * rij[2];
      double udot = ui[0] * uj[0] + ui[1] * uj[1] + ui[2] * uj[2];
      double denom = 1.0 - udot * udot;
      double lambda, mu;
      const double SMALL = 1.0e-14;

      if (denom < SMALL) {
        if (uidot != 0.0) {
          lambda = std::copysign(Hi, uidot);
          mu = lambda * udot - ujdot;
          if (std::fabs(mu) > Hj) mu = std::copysign(Hj, mu);
        } else {
          lambda = 0.0;
          mu = 0.0;
        }
      } else {
        double oden = 1.0 / denom;
        lambda = (uidot - udot * ujdot) * oden;
        mu = (-ujdot + udot * uidot) * oden;
        double vi = std::fabs(lambda) - Hi;
        double vj = std::fabs(mu) - Hj;
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

      double ci[3] = {x[i][0] + lambda * ui[0], x[i][1] + lambda * ui[1], x[i][2] + lambda * ui[2]};
      double cj[3] = {x[j][0] + mu * uj[0], x[j][1] + mu * uj[1], x[j][2] + mu * uj[2]};
      double del[3] = {ci[0] - cj[0], ci[1] - cj[1], ci[2] - cj[2]};
      double rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];

      uint64_t key = pair_key(tag[i], tag[j]);
      if (rsq < radsum * radsum) {
        seen_this_step.insert(key);
        if (active_contacts.find(key) == active_contacts.end()) {
          active_contacts.insert(key);
          const bool owns_contact =
            force->newton_pair || j < nlocal || tag[i] < tag[j];
          if (owns_contact) new_events_this_step++;
        }
      }
    }
  }

  for (auto it = active_contacts.begin(); it != active_contacts.end();) {
    if (seen_this_step.find(*it) == seen_this_step.end()) {
      it = active_contacts.erase(it);
    } else {
      ++it;
    }
  }

  events_cum += new_events_this_step;

  const long long step = static_cast<long long>(update->ntimestep);
  const bool at_interval = (step % log_every == 0);
  const bool at_end = (step == static_cast<long long>(update->laststep));
  if (at_interval || at_end) maybe_log(step, true);
}

/* ---------------------------------------------------------------------- */

void FixCountCollisions::maybe_log(long long step, bool /*force*/)
{
  if (step == last_logged_step) return;

  long long global_cum = 0;
  MPI_Allreduce(&events_cum, &global_cum, 1, MPI_LONG_LONG, MPI_SUM, world);
  long long new_since_last = global_cum - events_cum_prev;

  if (comm->me == 0 && fp) {
    fprintf(fp, "%lld %lld %lld\n", step, new_since_last, global_cum);
    fflush(fp);
  }

  events_cum_prev = global_cum;
  last_logged_step = step;
}
