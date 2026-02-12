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

class PairGranSpherocylHistory : public PairGranHookeHistory {
 public:
  PairGranSpherocylHistory(class LAMMPS *);
  ~PairGranSpherocylHistory() override;

  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  // Per-type spherocylinder geometry:
  // R[type] = radius
  // H[type] = half-length of cylindrical section (excluding caps)
  double *R;
  double *H;

  void allocate_sc();

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
};

}  // namespace LAMMPS_NS

#endif
#endif

