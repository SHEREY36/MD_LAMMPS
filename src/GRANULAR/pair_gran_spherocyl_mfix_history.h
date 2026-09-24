/* -*- c++ -*- ----------------------------------------------------------
   MFIX-matching spherocylinder DEM contact:
   - spherocylinder closest-approach geometry
   - Hertzian normal force ~ delta^(3/2)
   - MFIX-style normal damping ~ delta^(1/4) * v_n
   - no tangential force in the active force law
   Restitution calibration (head-on contact, exact to 1e-9 numerically):
     F_n = kn d^(3/2) + eta d^(1/4) v_n  gives a velocity-independent
     restitution e when eta = sqrt(5) sqrt(kn m*) |ln e| / sqrt(pi^2 + ln^2 e),
     m* = mi mj/(mi+mj).  The linear-spring prefactor 2 (used by MFIX) gives
     e = 0.54 for a target of 0.50.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(gran/spherocyl/mfix/history,PairGranSpherocylMfixHistory);
// clang-format on
#else

#ifndef LMP_PAIR_GRAN_SPHEROCYL_MFIX_HISTORY_H
#define LMP_PAIR_GRAN_SPHEROCYL_MFIX_HISTORY_H

#include "pair_gran_spherocyl_history.h"

namespace LAMMPS_NS {

class PairGranSpherocylMfixHistory : public PairGranSpherocylHistory {
 public:
  PairGranSpherocylMfixHistory(class LAMMPS *);
  void compute(int, int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
