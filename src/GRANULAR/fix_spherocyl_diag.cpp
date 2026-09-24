/* ----------------------------------------------------------------------
   fix spherocyl/diag  (see fix_spherocyl_diag.h for the command syntax)

   Design notes
   ------------
   * Continuous estimators are accumulated EVERY step and divided by the
     window time, so they are exact time averages (no sampling aliasing of
     soft contacts that last only tens of steps):
       - kinetic stress  (end_of_step, full-step velocities)
     (rotational statistics are sampled every `sample` steps, time weighted)
       - collisional stress sum (x_i-x_j) (x) F  (pair style, force stage)
       - shear work and non-conservative contact work (energy bookkeeping)
   * Collision-by-collision estimators come from per-contact records kept in
     the neighbor-history slots of the pair style (contact start = first step
     with overlap, contact end = first step without overlap).  The integral
     of (x_i-x_j) (x) F over each contact gives a second, independent
     collisional-stress estimator (it should agree with the continuous one
     apart from collisions straddling window boundaries).
   * Energy bookkeeping (exact for Newton + Lees-Edwards):
       d/dt [E_tr,pec + E_rot + U_el] = -V G:P  + W_nc
     where W_nc is the power of all non-elastic contact forces evaluated
     with the contact-point relative velocity.  The residual of this balance
     is a runaway / broken-physics sensor that works for elastic AND
     dissipative systems.
   * Peculiar momentum sum_i m c_i is conserved exactly (pairwise forces,
     sum m v_y = 0), so any drift flags a broken Lees-Edwards setup.
------------------------------------------------------------------------- */

#include "fix_spherocyl_diag.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "math_const.h"
#include "math_eigen.h"
#include "math_extra.h"
#include "memory.h"
#include "pair_gran_spherocyl_history.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_2PI;

namespace {
// density modes (in lamda = reduced, box-following coordinates)
const int MODES[10][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0}, {0, 2, 0},
                          {0, 0, 2}, {1, 1, 0}, {1, -1, 0}, {0, 1, 1}, {1, 0, 1}};
// x-momentum modes (shear banding): along gradient (y) and vorticity (z)
const int JXMODES[3][3] = {{0, 1, 0}, {0, 2, 0}, {0, 0, 1}};
// kinetic-energy (temperature) modes
const int EMODES[2][3] = {{0, 1, 0}, {0, 0, 1}};
const int GRIDS[3] = {2, 4, 8};

enum { W_ENERGY = 0, W_MOMENTUM, W_HARD, W_RES, W_MULTI, W_CLUSTER, W_GAIN, W_RUNAWAY };
}    // namespace

/* ---------------------------------------------------------------------- */

FixSpherocylDiag::FixSpherocylDiag(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), pa(nullptr), prefix(nullptr), fp_stress(nullptr), fp_energy(nullptr),
    fp_coll(nullptr), fp_struct(nullptr), fp_sensor(nullptr), fp_events(nullptr), pair(nullptr),
    Rtype(nullptr), Htype(nullptr)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, "fix spherocyl/diag", error);

  nwindow = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nwindow <= 0) error->all(FLERR, "Fix spherocyl/diag Nwindow must be > 0");
  prefix = utils::strdup(arg[4]);

  nsample = MAX(1, nwindow / 20);
  log_fraction = 0.0;
  dur_min_thresh = 15.0;
  dur_long_thresh = 1.0e300;
  appendflag = 0;

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "sample") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix spherocyl/diag sample", error);
      nsample = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (nsample <= 0) error->all(FLERR, "Fix spherocyl/diag sample must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "log_fraction") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix spherocyl/diag log_fraction", error);
      log_fraction = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (log_fraction < 0.0 || log_fraction > 1.0)
        error->all(FLERR, "Fix spherocyl/diag log_fraction must be in [0,1]");
      iarg += 2;
    } else if (strcmp(arg[iarg], "dur_min") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix spherocyl/diag dur_min", error);
      dur_min_thresh = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "append") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix spherocyl/diag append", error);
      appendflag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown fix spherocyl/diag keyword: {}", arg[iarg]);
  }

  nevery = 1;
  vector_flag = 1;
  size_vector = NVEC;
  global_freq = 1;
  extvector = 0;
  peratom_flag = 1;
  size_peratom_cols = NPA;
  peratom_freq = 1;
  comm_forward = 1;

  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  for (int i = 0; i < atom->nlocal; i++) set_arrays(i);

  memset(acc, 0, sizeof(acc));
  memset(racc, 0, sizeof(racc));
  memset(rdurhist, 0, sizeof(rdurhist));
  memset(durhist, 0, sizeof(durhist));
  memset(sacc, 0, sizeof(sacc));
  memset(vec, 0, sizeof(vec));
  memset(warn_count, 0, sizeof(warn_count));
  for (int k = 0; k < NMIN; k++) accmin[k] = raccmin[k] = 1.0e300;
  for (int k = 0; k < 6; k++) G[k] = 0.0;
  uel_step = 0.0;
  t_win = 0.0;
  t_since_rot = 0.0;
  t_run = 0.0;
  rtime = 0.0;
  nsteps_win = 0;
  nsamp = 0;
  strain_total = 0.0;
  e_prev = 0.0;
  e_prev_valid = false;
  wshk_rate_prev = 0.0;
  pc0[0] = pc0[1] = pc0[2] = 0.0;

  if (comm->me == 0) open_files();
}

/* ---------------------------------------------------------------------- */

FixSpherocylDiag::~FixSpherocylDiag()
{
  if (atom) atom->delete_callback(id, Atom::GROW);
  memory->destroy(pa);
  // detach from the pair style only if it is still the object we registered with
  if (force && force->pair && pair && (force->pair == (Pair *) pair) && pair->diag == this)
    pair->diag = nullptr;
  delete[] prefix;
  for (FILE *fp : {fp_stress, fp_energy, fp_coll, fp_struct, fp_sensor, fp_events})
    if (fp) fclose(fp);
}

/* ---------------------------------------------------------------------- */

int FixSpherocylDiag::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSpherocylDiag::init()
{
  pair = dynamic_cast<PairGranSpherocylHistory *>(force->pair);
  if (!pair)
    error->all(FLERR,
               "Fix spherocyl/diag requires pair style gran/spherocyl/history or "
               "gran/spherocyl/mfix/history");
  pair->diag = this;

  int dim = 0;
  Rtype = (double *) pair->extract("spherocyl_R", dim);
  Htype = (double *) pair->extract("spherocyl_H", dim);
  if (!Rtype || !Htype) error->all(FLERR, "Fix spherocyl/diag could not extract R/H from pair style");

  if (!dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid")))
    error->all(FLERR, "Fix spherocyl/diag requires atom style ellipsoid");
  if (force->newton_pair == 0 && comm->ghost_velocity == 0)
    error->all(FLERR, "Fix spherocyl/diag requires comm_modify vel yes");
}

/* ----------------------------------------------------------------------
   start of every run: reset window and run accumulators, record the
   reference energy/momentum (the setup force computation has been done)
------------------------------------------------------------------------- */

void FixSpherocylDiag::setup(int /*vflag*/)
{
  reset_window();
  memset(racc, 0, sizeof(racc));
  memset(rdurhist, 0, sizeof(rdurhist));
  for (int k = 0; k < NMIN; k++) raccmin[k] = 1.0e300;
  rtime = 0.0;

  double etr, erot, pc[3], cmax2, c2m;
  instantaneous(etr, erot, pc, cmax2, c2m);
  double uel = 0.0;
  MPI_Allreduce(&uel_step, &uel, 1, MPI_DOUBLE, MPI_SUM, world);
  e_prev = etr + erot + uel;
  e_prev_valid = true;

  // kinetic shear-work rate at the start of the run (trapezoid rule)
  {
    MathExtra::multiply_shape_shape(domain->h_rate, domain->h_inv, G);
    double K[6] = {0, 0, 0, 0, 0, 0}, Kg[6], u[3], c[3];
    for (int i = 0; i < atom->nlocal; i++) {
      if (!(atom->mask[i] & groupbit)) continue;
      const double m = atom->rmass_flag ? atom->rmass[i] : atom->mass[atom->type[i]];
      stream_velocity(atom->x[i], u);
      for (int k = 0; k < 3; k++) c[k] = atom->v[i][k] - u[k];
      K[0] += m * c[0] * c[0];
      K[1] += m * c[1] * c[1];
      K[2] += m * c[2] * c[2];
      K[3] += m * c[0] * c[1];
      K[4] += m * c[0] * c[2];
      K[5] += m * c[1] * c[2];
    }
    MPI_Allreduce(K, Kg, 6, MPI_DOUBLE, MPI_SUM, world);
    // stored per rank as the local share so that the per-rank trapezoid sums add up
    double Kl[6];
    for (int k = 0; k < 6; k++) Kl[k] = K[k];
    wshk_rate_prev = -(G[0] * Kl[0] + G[1] * Kl[1] + G[2] * Kl[2] + G[5] * Kl[3] + G[4] * Kl[4] +
                       G[3] * Kl[5]);
  }
  for (int k = 0; k < 3; k++) pc0[k] = pc[k];
  memset(warn_count, 0, sizeof(warn_count));
}

/* ---------------------------------------------------------------------- */

void FixSpherocylDiag::reset_window()
{
  memset(acc, 0, sizeof(acc));
  for (int k = 0; k < NMIN; k++) accmin[k] = 1.0e300;
  memset(sacc, 0, sizeof(sacc));
  nsamp = 0;
  t_win = 0.0;
  t_since_rot = 0.0;
  nsteps_win = 0;
  evbuf.clear();
  memset(durhist, 0, sizeof(durhist));
}

/* ----------------------------------------------------------------------
   velocity gradient and streaming velocity, identical to compute temp/deform
------------------------------------------------------------------------- */

void FixSpherocylDiag::stream_velocity(const double *x, double *u)
{
  double lamda[3];
  const double *h_rate = domain->h_rate;
  const double *h_ratelo = domain->h_ratelo;
  domain->x2lamda(const_cast<double *>(x), lamda);
  u[0] = h_rate[0] * lamda[0] + h_rate[5] * lamda[1] + h_rate[4] * lamda[2] + h_ratelo[0];
  u[1] = h_rate[1] * lamda[1] + h_rate[3] * lamda[2] + h_ratelo[1];
  u[2] = h_rate[2] * lamda[2] + h_ratelo[2];
}

/* ---------------------------------------------------------------------- */

void FixSpherocylDiag::atom_omega(int i, double *omega, double *inertia)
{
  auto *avec = dynamic_cast<AtomVecEllipsoid *>(atom->avec);
  const int itype = atom->type[i];
  const double mass = atom->rmass_flag ? atom->rmass[i] : atom->mass[itype];
  PairGranSpherocylHistory::spherocyl_inertia(mass, Rtype[itype], Htype[itype], inertia);
  const int e = atom->ellipsoid[i];
  if (e < 0) {
    omega[0] = omega[1] = omega[2] = 0.0;
    return;
  }
  MathExtra::mq_to_omega(atom->angmom[i], avec->bonus[e].quat, inertia, omega);
}

/* ----------------------------------------------------------------------
   called by the pair style at the start of every force computation
------------------------------------------------------------------------- */

void FixSpherocylDiag::pair_begin()
{
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    pa[i][PA_ZPREV] = pa[i][PA_ZCUR];
    pa[i][PA_ZCUR] = 0.0;
    for (int k = 0; k < 6; k++) pa[i][PA_FNC + k] = 0.0;
  }
  comm->forward_comm(this);    // ghost atoms need their previous-step contact count
  uel_step = 0.0;
  MathExtra::multiply_shape_shape(domain->h_rate, domain->h_inv, G);
}

/* ---------------------------------------------------------------------- */

void FixSpherocylDiag::contact_start_atoms(int i, int j, int nlocal, double tnow)
{
  const int both[2] = {i, j};
  for (int a : both) {
    if (a >= nlocal) continue;
    pa[a][PA_NCW] += 1.0;
    pa[a][PA_NCT] += 1.0;
    if (pa[a][PA_LASTEND] > -1.0e299) {
      const double ff = tnow - pa[a][PA_LASTEND];
      if (ff >= 0.0) {
        acc[A_FFN] += 1.0;
        acc[A_FFS] += ff;
        acc[A_FFS2] += ff * ff;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixSpherocylDiag::contact_end_atoms(int i, int j, int nlocal, double tnow)
{
  tagint *tag = atom->tag;
  if (i < nlocal) {
    pa[i][PA_LASTP] = (double) tag[j];
    pa[i][PA_LASTEND] = tnow;
  }
  if (j < nlocal) {
    pa[j][PA_LASTP] = (double) tag[i];
    pa[j][PA_LASTEND] = tnow;
  }
}

/* ---------------------------------------------------------------------- */

bool FixSpherocylDiag::sample_pair(tagint a, tagint b) const
{
  if (log_fraction <= 0.0) return false;
  if (log_fraction >= 1.0) return true;
  const uint64_t lo = (uint64_t) MIN(a, b), hi = (uint64_t) MAX(a, b);
  uint64_t hsh = lo * 0x9E3779B97F4A7C15ULL ^ (hi + 0x632BE59BD9B4E019ULL + (lo << 6) + (lo >> 2));
  hsh ^= hsh >> 33;
  hsh *= 0xff51afd7ed558ccdULL;
  hsh ^= hsh >> 33;
  return (double) (hsh >> 11) * (1.0 / 9007199254740992.0) < log_fraction;
}

void FixSpherocylDiag::push_event(const double *rec)
{
  evbuf.insert(evbuf.end(), rec, rec + NEV);
}

void FixSpherocylDiag::add_duration(double steps)
{
  int k = (int) floor(20.0 * log10(MAX(steps, 1.0)));
  if (k < 0) k = 0;
  if (k >= NDH) k = NDH - 1;
  durhist[k] += 1.0;
}

/* quantile q of a log-binned duration histogram (steps), log-linear interpolation */

double FixSpherocylDiag::hist_quantile(const double *h, double q)
{
  double tot = 0.0;
  for (int k = 0; k < NDH; k++) tot += h[k];
  if (tot <= 0.0) return 0.0;
  const double target = q * tot;
  double cum = 0.0;
  for (int k = 0; k < NDH; k++) {
    if (cum + h[k] >= target && h[k] > 0.0) {
      const double f = (target - cum) / h[k];
      return pow(10.0, (k + f) / 20.0);
    }
    cum += h[k];
  }
  return pow(10.0, NDH / 20.0);
}

/* ----------------------------------------------------------------------
   every step: kinetic stress, rotational tensor, coordination histogram
------------------------------------------------------------------------- */

void FixSpherocylDiag::end_of_step()
{
  const double dt = update->dt;
  double **v = atom->v;
  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  const int nlocal = atom->nlocal;

  MathExtra::multiply_shape_shape(domain->h_rate, domain->h_inv, G);

  // every step (exact time integrals): kinetic tensor, (m c^2)^2, coordination
  double K[6] = {0, 0, 0, 0, 0, 0};
  double c4 = 0.0, pnc = 0.0;
  double zh[4] = {0, 0, 0, 0};
  double u[3], c[3];

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const double m = atom->rmass_flag ? atom->rmass[i] : atom->mass[type[i]];
    stream_velocity(x[i], u);
    c[0] = v[i][0] - u[0];
    c[1] = v[i][1] - u[1];
    c[2] = v[i][2] - u[2];
    K[0] += m * c[0] * c[0];
    K[1] += m * c[1] * c[1];
    K[2] += m * c[2] * c[2];
    K[3] += m * c[0] * c[1];
    K[4] += m * c[0] * c[2];
    K[5] += m * c[1] * c[2];
    const double mc2 = m * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    c4 += mc2 * mc2;
    const int z = (int) (pa[i][PA_ZCUR] + 0.5);
    zh[MIN(z, 3)] += 1.0;

    // non-elastic contact power with full-step (time-centred) velocities: this
    // matches the velocity-Verlet energy change to O(dt^2)
    const double *fnc = &pa[i][PA_FNC];
    const double *tnc = &pa[i][PA_TNC];
    if (fnc[0] != 0.0 || fnc[1] != 0.0 || fnc[2] != 0.0 || tnc[0] != 0.0 || tnc[1] != 0.0 ||
        tnc[2] != 0.0) {
      double om[3], inertia[3];
      atom_omega(i, om, inertia);
      pnc += fnc[0] * c[0] + fnc[1] * c[1] + fnc[2] * c[2] + tnc[0] * om[0] + tnc[1] * om[1] +
          tnc[2] * om[2];
    }
  }

  acc[A_WNC] += pnc * dt;
  for (int k = 0; k < 6; k++) acc[A_K + k] += K[k] * dt;
  for (int k = 0; k < 4; k++) acc[A_Z + k] += zh[k];
  acc[A_C4] += c4 * dt;
  // kinetic shear work rate -sum_ab G_ab K_ab (K symmetric, G upper triangular),
  // integrated with the trapezoid rule (second order in dt)
  const double wrate =
      -(G[0] * K[0] + G[1] * K[1] + G[2] * K[2] + G[5] * K[3] + G[4] * K[4] + G[3] * K[5]);
  acc[A_WSHK] += 0.5 * (wshk_rate_prev + wrate) * dt;
  wshk_rate_prev = wrate;

  t_win += dt;
  t_since_rot += dt;
  nsteps_win++;
  strain_total += G[5] * dt;

  // windows and samples are counted from the start of each run, so every
  // run (block) consists of whole windows whatever the global timestep is
  if (nsteps_win % nsample == 0) {
    rotation_sample(t_since_rot);
    t_since_rot = 0.0;
    structure_sample();
  }
  if (nsteps_win >= nwindow) {
    if (t_since_rot > 0.0) rotation_sample(t_since_rot);
    t_since_rot = 0.0;
    finish_window();
  }
}

/* ----------------------------------------------------------------------
   rotational statistics, sampled with weight w (time since last sample)
------------------------------------------------------------------------- */

void FixSpherocylDiag::rotation_sample(double w)
{
  double **angmom = atom->angmom;
  int *mask = atom->mask;
  const int nlocal = atom->nlocal;
  double R[6] = {0, 0, 0, 0, 0, 0}, W[3] = {0, 0, 0}, w4 = 0.0;
  double om[3], inertia[3];
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    atom_omega(i, om, inertia);
    const double *L = angmom[i];
    R[0] += L[0] * om[0];
    R[1] += L[1] * om[1];
    R[2] += L[2] * om[2];
    R[3] += 0.5 * (L[0] * om[1] + L[1] * om[0]);
    R[4] += 0.5 * (L[0] * om[2] + L[2] * om[0]);
    R[5] += 0.5 * (L[1] * om[2] + L[2] * om[1]);
    W[0] += om[0];
    W[1] += om[1];
    W[2] += om[2];
    const double lw = L[0] * om[0] + L[1] * om[1] + L[2] * om[2];
    w4 += lw * lw;
  }
  for (int k = 0; k < 6; k++) acc[A_ROT + k] += R[k] * w;
  for (int k = 0; k < 3; k++) acc[A_SPIN + k] += W[k] * w;
  acc[A_W4] += w4 * w;
}

/* ---------------------------------------------------------------------- */

void FixSpherocylDiag::post_run()
{
  // flush a partial window so every run's data is self contained
  if (nsteps_win > 0) {
    if (t_since_rot > 0.0) rotation_sample(t_since_rot);
    t_since_rot = 0.0;
    finish_window();
  }
}

/* ----------------------------------------------------------------------
   instantaneous peculiar translational energy, rotational energy,
   peculiar momentum, max |c|^2 and mean |c|^2 (all global)
------------------------------------------------------------------------- */

void FixSpherocylDiag::instantaneous(double &etr, double &erot, double *pc, double &cmax2, double &c2m)
{
  double **v = atom->v;
  double **x = atom->x;
  double **angmom = atom->angmom;
  int *mask = atom->mask;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  double loc[6] = {0, 0, 0, 0, 0, 0};    // etr erot pcx pcy pcz sum c^2
  double lmax = 0.0, u[3], c[3], w[3], inertia[3];
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const double m = atom->rmass_flag ? atom->rmass[i] : atom->mass[type[i]];
    stream_velocity(x[i], u);
    for (int k = 0; k < 3; k++) c[k] = v[i][k] - u[k];
    const double c2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
    loc[0] += 0.5 * m * c2;
    atom_omega(i, w, inertia);
    loc[1] += 0.5 * (angmom[i][0] * w[0] + angmom[i][1] * w[1] + angmom[i][2] * w[2]);
    loc[2] += m * c[0];
    loc[3] += m * c[1];
    loc[4] += m * c[2];
    loc[5] += c2;
    lmax = MAX(lmax, c2);
  }
  double glob[6];
  MPI_Allreduce(loc, glob, 6, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&lmax, &cmax2, 1, MPI_DOUBLE, MPI_MAX, world);
  const double ng = MAX(1.0, (double) group->count(igroup));
  etr = glob[0];
  erot = glob[1];
  pc[0] = glob[2];
  pc[1] = glob[3];
  pc[2] = glob[4];
  c2m = glob[5] / ng;
}

/* ----------------------------------------------------------------------
   structure sample: S(k), momentum/energy modes, density dispersion, Q
------------------------------------------------------------------------- */

void FixSpherocylDiag::structure_sample()
{
  double **v = atom->v;
  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  auto *avec = dynamic_cast<AtomVecEllipsoid *>(atom->avec);

  // packed local sums
  // [0..19]  density modes cos/sin
  // [20..25] jx modes cos/sin
  // [26..29] energy modes e*cos, e*sin
  // [30]     sum e, [31] sum e^2, [32] sum cx^2
  // [33..38] sum u_a u_b
  const int NL = 39;
  double loc[NL], glob[NL];
  memset(loc, 0, sizeof(loc));
  const int ncell = 8 + 64 + 512;
  std::vector<int> cloc(ncell, 0), cglob(ncell, 0);

  double lamda[3], u[3], c[3], a[3][3];
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const double m = atom->rmass_flag ? atom->rmass[i] : atom->mass[type[i]];
    domain->x2lamda(x[i], lamda);
    stream_velocity(x[i], u);
    for (int k = 0; k < 3; k++) c[k] = v[i][k] - u[k];
    const double e = 0.5 * m * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);

    for (int mm = 0; mm < NMODE; mm++) {
      const double ph = MY_2PI *
          (MODES[mm][0] * lamda[0] + MODES[mm][1] * lamda[1] + MODES[mm][2] * lamda[2]);
      loc[2 * mm] += cos(ph);
      loc[2 * mm + 1] += sin(ph);
    }
    for (int mm = 0; mm < NJX; mm++) {
      const double ph = MY_2PI *
          (JXMODES[mm][0] * lamda[0] + JXMODES[mm][1] * lamda[1] + JXMODES[mm][2] * lamda[2]);
      loc[20 + 2 * mm] += c[0] * cos(ph);
      loc[20 + 2 * mm + 1] += c[0] * sin(ph);
    }
    for (int mm = 0; mm < NEM; mm++) {
      const double ph = MY_2PI *
          (EMODES[mm][0] * lamda[0] + EMODES[mm][1] * lamda[1] + EMODES[mm][2] * lamda[2]);
      loc[26 + 2 * mm] += e * cos(ph);
      loc[26 + 2 * mm + 1] += e * sin(ph);
    }
    loc[30] += e;
    loc[31] += e * e;
    loc[32] += c[0] * c[0];

    const int ie = atom->ellipsoid[i];
    if (ie >= 0) {
      MathExtra::quat_to_mat(avec->bonus[ie].quat, a);
      double ua[3] = {a[0][2], a[1][2], a[2][2]};
      const double inv = 1.0 / sqrt(ua[0] * ua[0] + ua[1] * ua[1] + ua[2] * ua[2]);
      for (double &q : ua) q *= inv;
      loc[33] += ua[0] * ua[0];
      loc[34] += ua[1] * ua[1];
      loc[35] += ua[2] * ua[2];
      loc[36] += ua[0] * ua[1];
      loc[37] += ua[0] * ua[2];
      loc[38] += ua[1] * ua[2];
    }

    int off = 0;
    for (int g : GRIDS) {
      int idx[3];
      for (int k = 0; k < 3; k++) {
        int b = (int) floor(lamda[k] * g);
        b %= g;
        if (b < 0) b += g;
        idx[k] = b;
      }
      cloc[off + (idx[2] * g + idx[1]) * g + idx[0]]++;
      off += g * g * g;
    }
  }

  MPI_Allreduce(loc, glob, NL, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(cloc.data(), cglob.data(), ncell, MPI_INT, MPI_SUM, world);

  const double N = MAX(1.0, (double) group->count(igroup));
  for (int mm = 0; mm < NMODE; mm++)
    sacc[S_SK + mm] += (glob[2 * mm] * glob[2 * mm] + glob[2 * mm + 1] * glob[2 * mm + 1]) / N;

  const double cx2 = glob[32] / N;
  for (int mm = 0; mm < NJX; mm++) {
    const double re = glob[20 + 2 * mm], im = glob[20 + 2 * mm + 1];
    if (cx2 > 0.0) sacc[S_JX + mm] += (re * re + im * im) / (N * cx2);
  }
  const double emean = glob[30] / N;
  const double evar = glob[31] / N - emean * emean;
  // energy modes use density modes (010) = index 1 and (001) = index 2
  const int dens_index[NEM] = {1, 2};
  for (int mm = 0; mm < NEM; mm++) {
    const int di = dens_index[mm];
    const double re = glob[26 + 2 * mm] - emean * glob[2 * di];
    const double im = glob[26 + 2 * mm + 1] - emean * glob[2 * di + 1];
    if (evar > 0.0) sacc[S_EM + mm] += (re * re + im * im) / (N * evar);
  }
  int off = 0;
  for (int gi = 0; gi < NGRID; gi++) {
    const int g = GRIDS[gi], nc = g * g * g;
    double s1 = 0.0, s2 = 0.0;
    for (int k = 0; k < nc; k++) {
      s1 += cglob[off + k];
      s2 += (double) cglob[off + k] * cglob[off + k];
    }
    const double mean = s1 / nc, var = s2 / nc - mean * mean;
    if (mean > 0.0) sacc[S_DISP + gi] += var / mean;
    off += nc;
  }
  for (int k = 0; k < 6; k++) sacc[S_Q + k] += glob[33 + k] / N - (k < 3 ? 1.0 / 3.0 : 0.0);
  nsamp++;
}

/* ----------------------------------------------------------------------
   finish an output window: reduce, write files, sensors, reset
------------------------------------------------------------------------- */

void FixSpherocylDiag::finish_window()
{
  if (t_win <= 0.0) {
    reset_window();
    return;
  }

  double g[NACC], gmin[NMIN], gdh[NDH];
  MPI_Allreduce(acc, g, NACC, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(accmin, gmin, NMIN, MPI_DOUBLE, MPI_MIN, world);
  MPI_Allreduce(durhist, gdh, NDH, MPI_DOUBLE, MPI_SUM, world);
  const double dur_med = hist_quantile(gdh, 0.5);
  const double dur_p10 = hist_quantile(gdh, 0.1);

  const double N = MAX(1.0, (double) group->count(igroup));
  const double V = domain->xprd * domain->yprd * domain->zprd;
  const double n = N / V;
  const double tw = t_win;
  const double dt = update->dt;
  const double tnow = update->atime + (update->ntimestep - update->atimestep) * dt;

  // instantaneous state at window end
  double etr, erot, pc[3], cmax2, c2m;
  instantaneous(etr, erot, pc, cmax2, c2m);
  double uel = 0.0;
  MPI_Allreduce(&uel_step, &uel, 1, MPI_DOUBLE, MPI_SUM, world);
  const double e_now = etr + erot + uel;

  // per-particle collision-count dispersion over this window
  double ncl[2] = {0.0, 0.0}, ncg[2];
  for (int i = 0; i < atom->nlocal; i++) {
    if (!(atom->mask[i] & groupbit)) continue;
    ncl[0] += pa[i][PA_NCW];
    ncl[1] += pa[i][PA_NCW] * pa[i][PA_NCW];
    pa[i][PA_NCW] = 0.0;
  }
  MPI_Allreduce(ncl, ncg, 2, MPI_DOUBLE, MPI_SUM, world);
  const double ncmean = ncg[0] / N, ncvar = ncg[1] / N - ncmean * ncmean;
  const double ncdisp = ncmean > 0.0 ? ncvar / ncmean : 0.0;

  // --- derived quantities ---
  double K[6], C[9], CE[9], Rt[6];
  for (int k = 0; k < 6; k++) K[k] = g[A_K + k] / (tw * V);
  for (int k = 0; k < 9; k++) {
    C[k] = g[A_C + k] / (tw * V);
    CE[k] = g[A_CE + k] / (tw * V);
  }
  for (int k = 0; k < 6; k++) Rt[k] = g[A_ROT + k] / (tw * N);
  const double Ttr = (g[A_K] + g[A_K + 1] + g[A_K + 2]) / (3.0 * N * tw);
  const double Trot = 0.5 * (g[A_ROT] + g[A_ROT + 1] + g[A_ROT + 2]) / (N * tw);
  const double a2tr = Ttr > 0.0 ? g[A_C4] / (N * tw) / (15.0 * Ttr * Ttr) - 1.0 : 0.0;
  const double a2rot = Trot > 0.0 ? g[A_W4] / (N * tw) / (8.0 * Trot * Trot) - 1.0 : 0.0;

  const double wsh = g[A_WSHK] + g[A_WSHC];
  const double wnc = g[A_WNC] + g[A_WNCG];
  const double dE = e_prev_valid ? e_now - e_prev : 0.0;
  const double resid = dE - wsh - wnc;
  // normalise by the work terms, or by 1e-3 of the total energy when there is
  // no work (elastic, unsheared runs): resid_rel = 1 then means dE/E = 1e-3
  const double scale = MAX(MAX(fabs(wsh), fabs(wnc)), MAX(1.0e-3 * fabs(e_now), 1.0e-300));
  const double resid_rel = resid / scale;

  const double mmean = [&] {
    double ml = 0.0, mg;
    for (int i = 0; i < atom->nlocal; i++)
      if (atom->mask[i] & groupbit)
        ml += atom->rmass_flag ? atom->rmass[i] : atom->mass[atom->type[i]];
    MPI_Allreduce(&ml, &mg, 1, MPI_DOUBLE, MPI_SUM, world);
    return mg / N;
  }();
  const double pnorm = sqrt(MAX(1.0e-300, (2.0 / 3.0) * mmean * etr));
  double dpc[3];
  for (int k = 0; k < 3; k++) dpc[k] = (pc[k] - pc0[k]) / pnorm;
  const double dpcmag = sqrt(dpc[0] * dpc[0] + dpc[1] * dpc[1] + dpc[2] * dpc[2]);
  const double cratio = c2m > 0.0 ? sqrt(cmax2 / c2m) : 0.0;

  const double nstart = g[A_NSTART], nend = g[A_NEND], nbin = g[A_NBIN];
  const double nu = 2.0 * nstart / (N * tw);
  auto safe = [](double a, double b) { return b > 0.0 ? a / b : 0.0; };
  // momentum-weighted restitution: -sum g_post / sum g_pre (robust for rods,
  // where g_pre -> 0 for rotation-driven contacts makes single ratios explode)
  const double etr_m = g[A_ETR2] < 0.0 ? -g[A_ETR] / g[A_ETR2] : 0.0;
  const double etr_out = safe(g[A_ETROUT], g[A_NETR]);
  const double ec_m = g[A_EC2] < 0.0 ? -g[A_EC] / g[A_EC2] : 0.0;
  const double ec_out = safe(g[A_ECOUT], g[A_NEC]);
  const double durmin = gmin[M_DURMIN] < 1.0e299 ? gmin[M_DURMIN] : 0.0;
  const double durmax = gmin[M_NEGDURMAX] < 1.0e299 ? -gmin[M_NEGDURMAX] : 0.0;
  const double dmaxmax = gmin[M_NEGDMAXMAX] < 1.0e299 ? -gmin[M_NEGDMAXMAX] : 0.0;
  const double ffm = safe(g[A_FFS], g[A_FFN]);
  const double ffcv2 = ffm > 0.0 ? safe(g[A_FFS2], g[A_FFN]) / (ffm * ffm) - 1.0 : 0.0;
  const double zsum = g[A_Z] + g[A_Z + 1] + g[A_Z + 2] + g[A_Z + 3];

  double S[NSACC];
  for (int k = 0; k < NSACC; k++) S[k] = nsamp > 0 ? sacc[k] / nsamp : 0.0;
  double smax = 0.0, savg = 0.0;
  for (int mm = 0; mm < NMODE; mm++) {
    smax = MAX(smax, S[S_SK + mm]);
    savg += S[S_SK + mm] / NMODE;
  }
  double lam_max = 0.0;
  if (nsamp > 0) {
    double Qm[3][3] = {{S[S_Q], S[S_Q + 3], S[S_Q + 4]},
                       {S[S_Q + 3], S[S_Q + 1], S[S_Q + 5]},
                       {S[S_Q + 4], S[S_Q + 5], S[S_Q + 2]}};
    double ev[3], evec[3][3];
    if (MathEigen::jacobi3(Qm, ev, evec) == 0) lam_max = MAX(ev[0], MAX(ev[1], ev[2]));
  }

  // --- write files (rank 0) ---
  if (comm->me == 0) {
    const double step = (double) update->ntimestep;
    fprintf(fp_stress, "%.0f %.10g %.8g %.8g %.0f %.10g %.10g %.10g", step, tnow, strain_total, tw,
            N, V, Ttr, Trot);
    for (double q : K) fprintf(fp_stress, " %.10g", q);
    for (double q : C) fprintf(fp_stress, " %.10g", q);
    for (double q : CE) fprintf(fp_stress, " %.10g", q);
    fprintf(fp_stress, " %.8g %.8g\n", a2tr, a2rot);
    fflush(fp_stress);

    fprintf(fp_energy,
            "%.0f %.10g %.8g %.12g %.12g %.12g %.10g %.10g %.10g %.10g %.10g %.6g %.6g %.4g %.4g "
            "%.4g %.6g",
            step, tnow, strain_total, etr, erot, uel, g[A_WSHK], g[A_WSHC], wnc, g[A_WPOS], dE,
            resid, resid_rel, dpc[0], dpc[1], dpc[2], cratio);
    for (int k = 0; k < 3; k++) fprintf(fp_energy, " %.8g", g[A_SPIN + k] / (N * tw));
    for (double q : Rt) fprintf(fp_energy, " %.8g", q);
    fprintf(fp_energy, "\n");
    fflush(fp_energy);

    fprintf(fp_coll,
            "%.0f %.10g %.8g %.0f %.8g %.0f %.6g %.8g %.6g %.8g %.6g %.8g %.8g %.6g %.6g %.6g %.8g "
            "%.6g %.6g %.8g %.8g %.6g %.8g %.8g %.8g %.6g %.6g %.6g %.6g %.8g %.6g %.6g %.6g %.6g %.6g "
            "%.6g",
            step, tnow, strain_total, nstart, nu, nend, safe(nbin, nend), etr_m, etr_out, ec_m, ec_out,
            safe(g[A_GN], nend), safe(g[A_GN2], nend), safe(g[A_DUR], nend), durmin, durmax,
            safe(g[A_DURT], nend), safe(g[A_NSHORT], nend), safe(g[A_NLONG], nend), safe(g[A_DMAX], nend), dmaxmax,
            safe(g[A_NANTI], nend), safe(g[A_DEFRAC], nend), safe(g[A_DETR], nend),
            safe(g[A_DEROT], nend), safe(g[A_NSAME], nend), safe(g[A_NTIP], nend),
            safe(g[A_UU], nend), safe(g[A_NMULTI], nend), ffm, ffcv2, ncdisp, safe(g[A_Z], zsum),
            safe(g[A_Z + 1], zsum), safe(g[A_Z + 2], zsum), safe(g[A_Z + 3], zsum));
    for (int k = 0; k < 6; k++) fprintf(fp_coll, " %.6g", safe(g[A_NN + k], nend));
    fprintf(fp_coll, " %.6g %.6g\n", dur_med, dur_p10);
    fflush(fp_coll);

    fprintf(fp_struct, "%.0f %.10g %.8g %d", step, tnow, strain_total, nsamp);
    for (int k = 0; k < NSACC; k++) fprintf(fp_struct, " %.6g", S[k]);
    fprintf(fp_struct, " %.6g %.6g\n", lam_max, 1.5 * lam_max);
    fflush(fp_struct);
  }

  // --- sampled collision events ---
  if (log_fraction > 0.0) {
    int nloc = (int) evbuf.size();
    std::vector<int> counts(comm->nprocs), displs(comm->nprocs);
    MPI_Gather(&nloc, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, world);
    std::vector<double> all;
    if (comm->me == 0) {
      int tot = 0;
      for (int p = 0; p < comm->nprocs; p++) {
        displs[p] = tot;
        tot += counts[p];
      }
      all.resize(MAX(tot, 1));
    }
    MPI_Gatherv(evbuf.data(), nloc, MPI_DOUBLE, all.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, world);
    if (comm->me == 0 && fp_events) {
      int tot = 0;
      for (int p = 0; p < comm->nprocs; p++) tot += counts[p];
      for (int r = 0; r + NEV <= tot; r += NEV) {
        const double *e = &all[r];
        fprintf(fp_events, "%.10g %.0f %.0f", e[0], e[1], e[2]);
        for (int k = 3; k < NEV; k++) fprintf(fp_events, " %.8g", e[k]);
        fprintf(fp_events, "\n");
      }
      fflush(fp_events);
    }
  }

  // --- sensors ---
  const bool dissipative = wnc < 0.0 && fabs(wnc) > 1.0e-6 * fabs(wsh);
  std::string flags;
  if (e_prev_valid && fabs(resid_rel) > 0.02) flags += "E";
  if (dpcmag > 1.0e-6) flags += "M";
  if (dmaxmax > 0.02) flags += "H";
  if (nend > 0 && safe(g[A_NSHORT], nend) > 0.01) flags += "R";
  if (nend > 0 && safe(g[A_NMULTI], nend) > 0.05) flags += "B";
  // an ideal gas has S(k) ~ Exp(1) per mode: use the mean over the 10 modes
  if (savg > 2.5 || S[S_DISP + 1] > 1.5) flags += "C";
  if (dissipative && nend > 0 && safe(g[A_NANTI], nend) > 0.01) flags += "G";
  if (cratio > 8.0) flags += "V";
  if (flags.empty()) flags = "-";
  if (comm->me == 0) {
    fprintf(fp_sensor, "%.0f %.10g %.8g %s %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g\n",
            (double) update->ntimestep, tnow, strain_total, flags.c_str(), resid_rel, dpcmag,
            dmaxmax, durmin, safe(g[A_NMULTI], nend), smax, S[S_DISP + 1],
            safe(g[A_NANTI], nend), cratio, savg, safe(g[A_NSHORT], nend));
    fflush(fp_sensor);
    if (flags != "-") {
      if (flags.find('E') != std::string::npos)
        warn(W_ENERGY, "energy bookkeeping residual > 2% of the work terms");
      if (flags.find('M') != std::string::npos)
        warn(W_MOMENTUM, "peculiar momentum is drifting (Lees-Edwards/forces broken?)");
      if (flags.find('H') != std::string::npos) warn(W_HARD, "max overlap > 2% of (Ri+Rj)");
      if (flags.find('R') != std::string::npos) warn(W_RES, "more than 1% of contacts resolved by fewer than dur_min steps");
      if (flags.find('B') != std::string::npos) warn(W_MULTI, "more than 5% multi-body collisions");
      if (flags.find('C') != std::string::npos) warn(W_CLUSTER, "density inhomogeneity (clustering?)");
      if (flags.find('G') != std::string::npos) warn(W_GAIN, "more than 1% of collisions gain energy");
      if (flags.find('V') != std::string::npos) warn(W_RUNAWAY, "runaway particle speed");
    }
  }

  // --- run-cumulative bookkeeping for input-script access ---
  for (int k = 0; k < NACC; k++) racc[k] += g[k];
  for (int k = 0; k < NDH; k++) rdurhist[k] += gdh[k];
  for (int k = 0; k < NMIN; k++) raccmin[k] = MIN(raccmin[k], gmin[k]);
  rtime += tw;
  t_run += tw;

  const double rn = racc[A_NEND];
  vec[0] = Ttr;
  vec[1] = Trot;
  vec[2] = nu;
  vec[3] = safe(racc[A_DMAX], rn);
  vec[4] = raccmin[M_NEGDMAXMAX] < 1.0e299 ? -raccmin[M_NEGDMAXMAX] : 0.0;
  vec[5] = safe(racc[A_DUR], rn);
  vec[6] = raccmin[M_DURMIN] < 1.0e299 ? raccmin[M_DURMIN] : 0.0;
  vec[7] = rtime > 0.0 ? (racc[A_K] + racc[A_K + 1] + racc[A_K + 2]) / (3.0 * N * rtime) : 0.0;
  vec[8] = rtime > 0.0 ? 0.5 * (racc[A_ROT] + racc[A_ROT + 1] + racc[A_ROT + 2]) / (N * rtime) : 0.0;
  vec[9] = resid_rel;
  vec[10] = smax;
  vec[11] = safe(racc[A_NMULTI], rn);
  vec[12] = racc[A_ETR2] < 0.0 ? -racc[A_ETR] / racc[A_ETR2] : 0.0;
  vec[13] = racc[A_EC2] < 0.0 ? -racc[A_EC] / racc[A_EC2] : 0.0;
  const double nT = n * Ttr;
  vec[14] = nT > 0.0 ? (K[0] + C[0]) / nT : 0.0;
  vec[15] = nT > 0.0 ? (K[1] + C[4]) / nT : 0.0;
  vec[16] = nT > 0.0 ? (K[2] + C[8]) / nT : 0.0;
  vec[17] = nT > 0.0 ? (K[3] + 0.5 * (C[1] + C[3])) / nT : 0.0;
  vec[18] = strain_total;
  vec[19] = rn;
  vec[20] = safe(racc[A_DURT], rn);
  vec[21] = safe(racc[A_NANTI], rn);
  // robust contact-time statistics of this run (dt is constant within a run)
  vec[22] = hist_quantile(rdurhist, 0.5) * dt;    // median contact duration (time)
  vec[23] = hist_quantile(rdurhist, 0.1) * dt;    // 10th percentile (time)
  vec[24] = hist_quantile(rdurhist, 0.5);         // median contact duration (steps)

  if (nend > 0) dur_long_thresh = 5.0 * dur_med;

  e_prev = e_now;
  e_prev_valid = true;
  reset_window();
}

/* ---------------------------------------------------------------------- */

void FixSpherocylDiag::warn(int which, const char *msg)
{
  if (warn_count[which] < 3)
    error->warning(FLERR, "fix spherocyl/diag step {}: {}", update->ntimestep, msg);
  else if (warn_count[which] == 3)
    error->warning(FLERR, "fix spherocyl/diag: further '{}' warnings suppressed this run", msg);
  warn_count[which]++;
}

/* ---------------------------------------------------------------------- */

double FixSpherocylDiag::compute_vector(int n)
{
  if (n < 0 || n >= NVEC) return 0.0;
  return vec[n];
}

/* ---------------------------------------------------------------------- */

void FixSpherocylDiag::open_files()
{
  const char *mode = appendflag ? "a" : "w";
  auto openf = [&](const char *ext) {
    std::string fname = std::string(prefix) + ext;
    FILE *fp = fopen(fname.c_str(), mode);
    if (!fp) error->one(FLERR, "Cannot open fix spherocyl/diag file {}: {}", fname, utils::getsyserror());
    return fp;
  };
  fp_stress = openf(".stress");
  fp_energy = openf(".energy");
  fp_coll = openf(".coll");
  fp_struct = openf(".struct");
  fp_sensor = openf(".sensors");
  if (log_fraction > 0.0) fp_events = openf(".events");
  if (appendflag) return;

  fprintf(fp_stress,
          "# fix spherocyl/diag: exact time averages over each window (every step accumulated)\n"
          "# K_ab = <sum m c_a c_b>/V, c = v - u(x) (temp/deform streaming profile)\n"
          "# C_ab = <sum_pairs (x_i-x_j)_a F_b>/V, centre-of-mass branch, branch index first "
          "(continuous)\n"
          "# CE_ab = same, from integrals over completed collisions (independent estimator)\n"
          "# P_ab = K_ab + C_ab ; reduced P*_ab = P_ab/(n T_tr), n = N/V\n"
          "# T_tr = <sum m c^2>/(3N) ; T_rot = <sum L.w>/(2N) (per rotational DOF of a smooth rod)\n"
          "# a2_tr = <(m c^2)^2>/(15 T_tr^2)-1 ; a2_rot = <(L.w)^2>/(8 T_rot^2)-1 (2 rot DOF)\n"
          "# 1:step 2:time 3:strain 4:t_win 5:N 6:V 7:T_tr 8:T_rot 9:Kxx 10:Kyy 11:Kzz 12:Kxy "
          "13:Kxz 14:Kyz 15:Cxx 16:Cxy 17:Cxz 18:Cyx 19:Cyy 20:Cyz 21:Czx 22:Czy 23:Czz 24:CExx "
          "25:CExy 26:CExz 27:CEyx 28:CEyy 29:CEyz 30:CEzx 31:CEzy 32:CEzz 33:a2_tr 34:a2_rot\n");
  fprintf(fp_energy,
          "# energy bookkeeping: dE = W_shear_kin + W_shear_col + W_nc + resid (per window)\n"
          "# E_tr = sum m c^2/2 (peculiar), E_rot = sum L.w/2, U_el = elastic contact energy\n"
          "# W_shear = -int V G:P dt ; W_nc = int sum (F - F_elastic).v_contact dt "
          "(dissipation <0), evaluated per atom with full-step velocities; W_nc_pos = positive part of the per-contact non-elastic power (approximate)\n"
          "# resid_rel = resid / max(|W_shear|, |W_nc|, 1e-3 E) ; dPc = change of peculiar momentum since run start / sqrt(N m T) (must stay ~0)\n"
          "# cmax/crms = max |c| / rms |c| (runaway detector); w = <omega>; R_ab = <sym(L_a "
          "w_b)>/N (R_xx+R_yy+R_zz = 2 T_rot)\n"
          "# 1:step 2:time 3:strain 4:E_tr 5:E_rot 6:U_el 7:W_shear_kin 8:W_shear_col 9:W_nc "
          "10:W_nc_pos 11:dE 12:resid 13:resid_rel 14:dPc_x 15:dPc_y 16:dPc_z 17:cmax/crms 18:wx "
          "19:wy 20:wz 21:Rxx 22:Ryy 23:Rzz 24:Rxy 25:Rxz 26:Ryz\n");
  fprintf(fp_coll,
          "# per-window collision statistics (contact = first overlapping step .. first "
          "separated step)\n"
          "# binary = no other contact on either body during the collision\n"
          "# e_tr = -g_n,post/g_n,pre with translational g = v_i-v_j projected on the collision "
          "impulse direction J/|J| ; e_c = same with the contact-point velocity (incl. w x rho)\n"
          "# dmax = max overlap/(Ri+Rj) ; gain = collisions whose pair energy increased by "
          ">0.1%%\n"
          "# dE/E = pair (relative translational + rotational) energy change / pre-collision "
          "energy\n"
          "# same = the particle's previous collision partner was the same body ; endcap = "
          "contact on a rod end (|lambda|=H)\n"
          "# freeflight: time between the end of a particle's collision and the start of its "
          "next (cv2 = 1 for a Poisson process)\n"
          "# collcount_disp = var/mean of per-particle collision counts in the window (1 = "
          "Poisson)\n"
          "# e_tr, e_c are momentum weighted: -sum g_n,post / sum g_n,pre over binary collisions; "
          "e_out = fraction of binary collisions whose individual e is outside [0,1]\n"
          "# long = contact longer than 5x the median duration of the previous window\n"
          "# 1:step 2:time 3:strain 4:N_start 5:nu 6:N_end 7:frac_binary 8:e_tr 9:e_tr_out 10:e_c "
          "11:e_c_out 12:<|g_n|> 13:<g_n^2> 14:dur_mean_steps 15:dur_min_steps 16:dur_max_steps "
          "17:tc_mean 18:frac_short 19:frac_long 20:dmax_mean 21:dmax_max 22:frac_gain "
          "23:dE/E_mean 24:dE_rel_tr_per_coll 25:dE_rot_per_coll 26:frac_same_partner "
          "27:frac_endcap 28:<|ui.uj|> 29:frac_multibody 30:freeflight_mean 31:freeflight_cv2 "
          "32:collcount_disp 33:P(z=0) 34:P(z=1) 35:P(z=2) 36:P(z>=3) 37:nn_xx 38:nn_yy 39:nn_zz "
          "40:nn_xy 41:nn_xz 42:nn_yz 43:dur_median_steps 44:dur_p10_steps\n");
  fprintf(fp_struct,
          "# structure samples averaged over the window; modes are in box-following (lamda) "
          "coordinates, k = 2 pi (n1 b1 + n2 b2 + n3 b3)\n"
          "# S(n) = |sum exp(i k.r)|^2/N (ideal gas ~1, clustering >> 1) ; Jx(n) = |sum c_x "
          "exp(ik.r)|^2/(N<c_x^2>) (shear banding) ; E(n) = kinetic-energy mode / (N var e)\n"
          "# D(g^3) = var/mean of particle counts in g^3 cells (Poisson = 1) ; Q_ab = <u_a u_b - "
          "delta_ab/3> ; S2 = 1.5 lambda_max (standard nematic order, 1 = aligned)\n"
          "# 1:step 2:time 3:strain 4:nsamp 5:S100 6:S010 7:S001 8:S200 9:S020 10:S002 11:S110 "
          "12:S1-10 13:S011 14:S101 15:Jx010 16:Jx020 17:Jx001 18:E010 19:E001 20:D2 21:D4 22:D8 "
          "23:Qxx 24:Qyy 25:Qzz 26:Qxy 27:Qxz 28:Qyz 29:lambda_max 30:S2\n");
  fprintf(fp_sensor,
          "# flags: E energy residual>2%%, M peculiar momentum drift, H max overlap>2%%, R "
          ">1%% of contacts shorter than dur_min steps, B >5%% multibody collisions, C clustering "
          "(mean of the 10 S(k) > 2.5 or D(4^3)>1.5), G >1%% energy-gaining collisions (dissipative runs), V "
          "runaway particle (|c|>8 c_rms), - = all OK\n"
          "# 1:step 2:time 3:strain 4:flags 5:resid_rel 6:dPc 7:dmax_max 8:dur_min_steps "
          "9:frac_multibody 10:S_max 11:D4 12:frac_gain 13:cmax/crms 14:S_mean 15:frac_short\n");
  if (fp_events)
    fprintf(fp_events,
            "# sampled collisions (all collisions of a hashed subset of particle pairs)\n"
            "# g = v_i - v_j (translational), gc = contact-point relative velocity, projected on "
            "the impulse direction J/|J| of the collision (j->i); g_t = |g_pre - (g_pre.J)J|\n"
            "# 1:time_end 2:tag_i 3:tag_j 4:dur_steps 5:g_n_pre 6:g_n_post 7:e_tr 8:gc_n_pre "
            "9:gc_n_post 10:e_c 11:|g_t_pre| 12:Erel_pre 13:Erel_post 14:Erot_pre 15:Erot_post "
            "16:W_nc 17:dmax/d 18:lever_min 19:lever_max 20:|ui.uj| 21:other_contacts_max "
            "22:same_partner\n");
  for (FILE *fp : {fp_stress, fp_energy, fp_coll, fp_struct, fp_sensor}) fflush(fp);
}

/* ----------------------------------------------------------------------
   per-atom array management
------------------------------------------------------------------------- */

void FixSpherocylDiag::grow_arrays(int nmax)
{
  memory->grow(pa, nmax, NPA, "spherocyl/diag:pa");
  array_atom = pa;
}

void FixSpherocylDiag::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int k = 0; k < NPA; k++) pa[j][k] = pa[i][k];
}

void FixSpherocylDiag::set_arrays(int i)
{
  for (int k = 0; k < NPA; k++) pa[i][k] = 0.0;
  pa[i][PA_LASTP] = -1.0;
  pa[i][PA_LASTEND] = -1.0e300;
}

int FixSpherocylDiag::pack_exchange(int i, double *buf)
{
  for (int k = 0; k < NPA; k++) buf[k] = pa[i][k];
  return NPA;
}

int FixSpherocylDiag::unpack_exchange(int nlocal, double *buf)
{
  for (int k = 0; k < NPA; k++) pa[nlocal][k] = buf[k];
  return NPA;
}

int FixSpherocylDiag::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                        int * /*pbc*/)
{
  for (int i = 0; i < n; i++) buf[i] = pa[list[i]][PA_ZPREV];
  return n;
}

void FixSpherocylDiag::unpack_forward_comm(int n, int first, double *buf)
{
  for (int i = 0; i < n; i++) pa[first + i][PA_ZPREV] = buf[i];
}

double FixSpherocylDiag::memory_usage()
{
  return (double) atom->nmax * NPA * sizeof(double) + evbuf.capacity() * sizeof(double);
}
