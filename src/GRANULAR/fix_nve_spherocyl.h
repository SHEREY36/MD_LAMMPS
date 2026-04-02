/* -*- c++ -*- ----------------------------------------------------------
   Spherocylinder NVE integrator:
   Uses atom_style ellipsoid storage (quat/angmom) with spherocylinder inertia.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(nve/spherocyl,FixNVESpherocyl);
// clang-format on
#else

#ifndef LMP_FIX_NVE_SPHEROCYL_H
#define LMP_FIX_NVE_SPHEROCYL_H

#include "fix_nve.h"

namespace LAMMPS_NS {

class FixNVESpherocyl : public FixNVE {
 public:
  FixNVESpherocyl(class LAMMPS *, int, char **);
  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;

 private:
  double dtq;
  class AtomVecEllipsoid *avec;
  class PairGranSpherocylHistory *pair_sc;
  double *R;
  double *H;
  int use_rot_sllod;
  double half_gdot_rot_sllod;
};

}    // namespace LAMMPS_NS

#endif
#endif
