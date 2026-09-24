/* ----------------------------------------------------------------------
   Spherocylinder NVE integrator for DEM:
   Time-integration uses spherocylinder principal moments while reusing
   atom_style ellipsoid quaternion/angmom storage.
------------------------------------------------------------------------- */

#include "fix_nve_spherocyl.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_extra.h"
#include "pair_gran_spherocyl_history.h"
#include "utils.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_PI;

namespace {
inline void spherocyl_inertia(double mass, double radius, double half_cyl, double *inertia)
{
  const double d = 2.0 * radius;
  const double lcyl = 2.0 * half_cyl;

  const double d2 = d * d;
  const double d3 = d2 * d;
  const double d4 = d2 * d2;
  const double d5 = d4 * d;
  const double lcyl2 = lcyl * lcyl;
  const double lcyl3 = lcyl2 * lcyl;

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

// Smooth spherocylinder model:
// remove angular-momentum component along the body symmetry axis.
// This enforces 2 rotational DOF (no spin about rod axis).
inline void remove_axial_spin(const double *quat, double *L)
{
  double a[3][3];
  MathExtra::quat_to_mat(quat, a);

  // body z-axis in space frame
  double u[3] = {a[0][2], a[1][2], a[2][2]};
  const double unorm = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + 1e-30);
  u[0] /= unorm;
  u[1] /= unorm;
  u[2] /= unorm;

  const double lpar = L[0] * u[0] + L[1] * u[1] + L[2] * u[2];
  L[0] -= lpar * u[0];
  L[1] -= lpar * u[1];
  L[2] -= lpar * u[2];
}
}    // namespace

/* ---------------------------------------------------------------------- */

FixNVESpherocyl::FixNVESpherocyl(LAMMPS *lmp, int narg, char **arg) : FixNVE(lmp, narg, arg)
{
  if (narg >= 4 && strcmp(arg[3], "rot_sllod") == 0)
    error->all(FLERR,
               "Fix nve/spherocyl: the rot_sllod option has been removed. In the lab frame "
               "used with fix deform remap v, a free rigid body obeys dL/dt = torque; "
               "Lees-Edwards images translate but do not rotate, so there is no rotational "
               "SLLOD term (it rotated every L about z and biased orientation statistics).");
  if (narg != 3) error->all(FLERR, "Illegal fix nve/spherocyl command");

  dtq = 0.0;
  avec = nullptr;
  pair_sc = nullptr;
  R = nullptr;
  H = nullptr;
}

/* ---------------------------------------------------------------------- */

void FixNVESpherocyl::init()
{
  avec = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  if (!avec) error->all(FLERR, "Fix nve/spherocyl requires atom style ellipsoid");

  int *ellipsoid = atom->ellipsoid;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (ellipsoid[i] < 0) error->one(FLERR, "Fix nve/spherocyl requires extended particles");

  if (!force->pair)
    error->all(FLERR, "Fix nve/spherocyl requires pair_style gran/spherocyl/history");

  pair_sc = dynamic_cast<PairGranSpherocylHistory *>(force->pair);
  if (!pair_sc)
    error->all(FLERR, "Fix nve/spherocyl requires pair_style gran/spherocyl/history (no hybrid)");

  int dim = 0;
  R = (double *) pair_sc->extract("spherocyl_R", dim);
  H = (double *) pair_sc->extract("spherocyl_H", dim);
  if (!R || !H)
    error->all(FLERR, "Fix nve/spherocyl could not extract spherocylinder geometry from pair style");

  FixNVE::init();
}

/* ---------------------------------------------------------------------- */

void FixNVESpherocyl::initial_integrate(int /*vflag*/)
{
  double dtfm;
  double inertia[3], omega[3];
  double *quat;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **angmom = atom->angmom;
  double **torque = atom->torque;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  dtq = 0.5 * dtv;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      const int itype = type[i];
      const double radius = R[itype];
      const double half_cyl = H[itype];
      if (radius <= 0.0 || half_cyl < 0.0)
        error->one(FLERR, i, "Fix nve/spherocyl found invalid geometry (R<=0 or H<0) for atom type");

      const double mass_i = rmass ? rmass[i] : mass[itype];
      dtfm = dtf / mass_i;
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];

      angmom[i][0] += dtf * torque[i][0];
      angmom[i][1] += dtf * torque[i][1];
      angmom[i][2] += dtf * torque[i][2];

      quat = bonus[ellipsoid[i]].quat;
      remove_axial_spin(quat, angmom[i]);

      spherocyl_inertia(mass_i, radius, half_cyl, inertia);

      MathExtra::mq_to_omega(angmom[i], quat, inertia, omega);
      MathExtra::richardson(quat, angmom[i], omega, inertia, dtq);
    }
}

/* ---------------------------------------------------------------------- */

void FixNVESpherocyl::final_integrate()
{
  double dtfm;
  double *quat;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **v = atom->v;
  double **f = atom->f;
  double **angmom = atom->angmom;
  double **torque = atom->torque;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      const double mass_i = rmass ? rmass[i] : mass[type[i]];
      dtfm = dtf / mass_i;
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];

      angmom[i][0] += dtf * torque[i][0];
      angmom[i][1] += dtf * torque[i][1];
      angmom[i][2] += dtf * torque[i][2];

      quat = bonus[ellipsoid[i]].quat;
      remove_axial_spin(quat, angmom[i]);
    }
}
