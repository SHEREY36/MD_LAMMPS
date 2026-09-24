/* ----------------------------------------------------------------------
   fix_count_collisions.h

   Standalone collision-event counter for spherocylinder DEM.

   Strategy
   --------
   Instead of relying on the FixNeighHistory touch flag (which is
   subject to neighbor-list-rebuild artefacts), this fix maintains its
   own persistent std::unordered_set of currently-active contact pairs,
   keyed by a canonical (tag_lo, tag_hi) identifier.

   At every timestep (post_force), it:
     1. Loops over the pair-style neighbor list.
     2. Recomputes the segment-segment contact geometry.
     3. Detects rising-edge contacts against its own active set.
     4. Logs step, new events since last log, cumulative events.

   It is measurement-only: no forces, torques, history, or atom state
   are modified.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(count/collisions,FixCountCollisions);
// clang-format on
#else

#ifndef LMP_FIX_COUNT_COLLISIONS_H
#define LMP_FIX_COUNT_COLLISIONS_H

#include "fix.h"

#include <cstdio>
#include <cstdint>
#include <unordered_set>

namespace LAMMPS_NS {

class FixCountCollisions : public Fix {
 public:
  FixCountCollisions(class LAMMPS *, int, char **);
  ~FixCountCollisions() override;

  int setmask() override;
  void init() override;
  void end_of_step() override;

 private:
  static inline uint64_t pair_key(tagint a, tagint b)
  {
    tagint lo = (a < b) ? a : b;
    tagint hi = (a < b) ? b : a;
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
  }

  std::unordered_set<uint64_t> active_contacts;
  std::unordered_set<uint64_t> seen_this_step;

  long long events_cum;
  long long events_cum_prev;
  int log_every;
  long long last_logged_step;
  FILE *fp;
  int counter_started;

  double *R;
  double *H;

  void scan_contacts();
  void maybe_log(long long step, bool force);
};

}    // namespace LAMMPS_NS

#endif
#endif
