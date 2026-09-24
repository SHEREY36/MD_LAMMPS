/* -*- c++ -*- ----------------------------------------------------------
   Spherocylinder DEM contact: Hooke + tangential history + friction
   Derived from PairGranHookeHistory (sphere) by replacing contact geometry.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(gran/spherocyl/history,PairGranSpherocylHistory);
// clang-format on
#else

#ifndef LMP_PAIR_GRAN_SPHEROCYL_HISTORY_H
#define LMP_PAIR_GRAN_SPHEROCYL_HISTORY_H

#include "pair_gran_hooke_history.h"

namespace LAMMPS_NS {

class FixSpherocylDiag;

class PairGranSpherocylHistory : public PairGranHookeHistory {
 public:
  PairGranSpherocylHistory(class LAMMPS *);
  ~PairGranSpherocylHistory() override;

  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;

  // Diagnostics hook, set by fix spherocyl/diag (nullptr = diagnostics off).
  FixSpherocylDiag *diag;

  // Neighbor-history layout per pair: [0..2] tangential shear displacement,
  // followed by NDIAG per-contact collision-record slots used by fix
  // spherocyl/diag.  FixNeighHistory negates all slots when the pair order
  // (i,j) is swapped after reneighboring.  Antisymmetric quantities (vectors
  // defined from j to i) are therefore stored as-is; symmetric quantities are
  // stored multiplied by the orientation marker D_MARK (+1 at creation), so
  // true value = stored * marker in either order.
  static constexpr int NSHEAR = 3;
  enum {
    D_MARK = 0,      // orientation marker (+1/-1)
    D_T0 = 1,        // S: simulation time at contact start
    D_G0 = 2,        // A: v_i - v_j at contact start (3)
    D_N0 = 5,        // A: unit normal (j->i) at contact start (3)
    D_GC0 = 8,       // S: contact-point relative normal velocity at start
    D_EROT0 = 9,     // S: rotational energy of the pair at start
    D_DMAX = 10,     // S: max overlap / (Ri+Rj)
    D_WNC = 11,      // S: non-conservative contact work
    D_WPOS = 12,     // S: positive part of the above
    D_RF = 13,       // S: integral of (x_i-x_j)_a F_b dt (9)
    D_J = 22,        // A: impulse on i (3)
    D_ZOTH = 25,     // S: max number of other contacts on i or j
    D_LMIN = 26,     // S: min(|lambda_i|/H_i, |mu_j|/H_j) at start
    D_LMAX = 27,     // S: max of the same
    D_UU = 28,       // S: |u_i . u_j| at start
    D_SAME = 29,     // S: 1 if i's previous collision partner was j
    D_VC0 = 30,      // A: contact-point relative velocity at start (3)
    NDIAG = 33
  };

  // Principal moments of a spherocylinder (capsule): body z = symmetry axis.
  static void spherocyl_inertia(double mass, double radius, double half_cyl, double *inertia);

 protected:
  // Per-type spherocylinder geometry:
  // R[type] = radius
  // H[type] = half-length of cylindrical section (excluding caps)
  double *R;
  double *H;

  // Lightweight contact-event counter:
  // event = pair transitions from non-contact to contact.
  long long events_new_local_step;
  long long events_new_global_step;
  long long events_cum_local;
  long long events_cum_global;
  long long events_cum_global_prev_log;
  long long events_last_logged_step;
  int events_log_every;
  FILE *events_fp;

  void allocate_sc();
  void maybe_log_collision_events();

  // Extract unit axis u from quaternion (atom_style ellipsoid)
  inline void axis_from_quat(const double *q, double *u) const;

  // Closest approach between 2 finite line segments centered at xi,xj
  // along unit axes ui,uj with half-lengths Hi,Hj (Vega-Lago / your MFIX kernel)
  // Outputs:
  //   del = Ci - Cj (vector from contact point on j to contact point on i)
  //   rhoi = Ci - xi  (lever arm)
  //   rhoj = Cj - xj  (lever arm)
  //   rsq = |del|^2
  inline void closest_approach(const double *xi, const double *xj,
                              const double *ui, const double *uj,
                              double Hi, double Hj,
                              double *del, double *rhoi, double *rhoj,
                              double &rsq) const;

  // angular velocity (space frame) and rotational energy of atom i
  void omega_erot(int i, double *omega, double &erot);

  // fix spherocyl/diag bookkeeping for a touching pair (called every step)
  //   F = total force on i, fel = magnitude of the elastic normal force,
  //   uel = elastic energy stored in this contact
  void diag_contact(int i, int j, int nlocal, int newcontact, double *hist, const double *del,
                    double r, double radsum, const double *rhoi, const double *rhoj,
                    const double *ui, const double *uj, const double *F, double fel, double uel);
  // fix spherocyl/diag bookkeeping when a touching pair separates
  void diag_release(int i, int j, int nlocal, double *hist, const double *rhoi,
                    const double *rhoj);
};

}  // namespace LAMMPS_NS

#endif
#endif
