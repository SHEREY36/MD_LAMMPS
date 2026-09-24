/* -*- c++ -*- ----------------------------------------------------------
   fix spherocyl/diag : stress decomposition, collision statistics,
   conservation sensors and clustering diagnostics for the
   gran/spherocyl/history and gran/spherocyl/mfix/history pair styles.

   Usage:
     fix ID group spherocyl/diag Nwindow prefix keyword value ...

     Nwindow  = steps per output window (all window quantities are exact
                time averages over the window, accumulated every step)
     prefix   = path prefix of the output files
                  prefix.stress   kinetic (6) + collisional (9, continuous)
                                  + collisional (9, collision-by-collision)
                  prefix.energy   energy bookkeeping, momentum, spin
                  prefix.coll     per-window collision statistics
                  prefix.struct   S(k), density dispersion, nematic Q
                  prefix.sensors  one line per window with warning flags
                  prefix.events   sampled per-collision records (optional)
     keywords:
       sample N      steps between structure samples (default Nwindow/20)
       log_fraction f   fraction of particle pairs whose collisions are
                        written to prefix.events (default 0 = off)
       dur_min N     contacts shorter than N steps are flagged (default 15)
       append yes/no append to existing files (default no)

   Stress conventions (V = box volume, averages over the window time):
     K_ab = (1/V) < sum_i m_i c_ia c_ib >,  c = v - u(x) (temp/deform bias)
     C_ab = (1/V) < sum_pairs (x_i - x_j)_a F_ij,b >  (branch index first,
            centre-of-mass branch vector, F_ij = force on i from j)
     Total P_ab = K_ab + C_ab.  Shear work rate = -V sum_ab G_ab P_ba with
     G_ab = du_a/dx_b.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(spherocyl/diag,FixSpherocylDiag);
// clang-format on
#else

#ifndef LMP_FIX_SPHEROCYL_DIAG_H
#define LMP_FIX_SPHEROCYL_DIAG_H

#include "fix.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixSpherocylDiag : public Fix {
  friend class PairGranSpherocylHistory;
  friend class PairGranSpherocylMfixHistory;

 public:
  FixSpherocylDiag(class LAMMPS *, int, char **);
  ~FixSpherocylDiag() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void end_of_step() override;
  void post_run() override;
  double compute_vector(int) override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  void set_arrays(int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double memory_usage() override;

  // ---- per-atom columns (also exported as f_ID[1..NPA]) ----
  // 0 contacts now, 1 contacts previous step, 2 tag of last partner, 3 time of
  // last contact end, 4 collisions this window, 5 collisions total,
  // 6-8 non-elastic contact force, 9-11 its torque (energy bookkeeping)
  enum { PA_ZCUR = 0, PA_ZPREV, PA_LASTP, PA_LASTEND, PA_NCW, PA_NCT, PA_FNC, PA_TNC = PA_FNC + 3, NPA = PA_TNC + 3 };

  // ---- window sum accumulators (local; MPI_SUM at window end) ----
  enum {
    A_K = 0,                     // 6: sum m c_a c_b dt  (xx yy zz xy xz yz)
    A_C = A_K + 6,               // 9: sum (x_i-x_j)_a F_b dt (continuous)
    A_CE = A_C + 9,              // 9: same, collision-by-collision
    A_ROT = A_CE + 9,            // 6: sum sym(L_a w_b) dt
    A_SPIN = A_ROT + 6,          // 3: sum w dt
    A_WSHK = A_SPIN + 3,         // shear work, kinetic part
    A_WSHC,                      // shear work, collisional part
    A_WNC,                       // non-elastic contact work, peculiar part (per atom, full-step velocities)
    A_WPOS,                      // positive part of the per-contact non-elastic power (approximate)
    A_WNCG,                      // non-elastic contact work, streaming part sum F_nc.(G r_ij)
    A_Z,                         // 4: atom-steps with z = 0,1,2,>=3
    A_C4 = A_Z + 4,              // sum (m c^2)^2 dt
    A_W4,                        // sum (L.w)^2 dt
    A_NSTART,                    // contacts started
    A_NEND,                      // contacts finished
    A_NBIN,                      // finished contacts that were binary
    A_ETR, A_ETR2, A_NETR,       // binary: sum g_n,post, sum g_n,pre, count (translational)
    A_EC, A_EC2, A_NEC,          // binary: same with the contact-point velocity
    A_ETROUT, A_ECOUT,           // binary collisions with e outside [0,1]
    A_NLONG,                     // contacts longer than dur_long_thresh
    A_GN, A_GN2,                 // pre-collision |g_n| moments (all)
    A_DUR,                       // sum of contact durations (steps)
    A_DURT,                      // sum of contact durations (time)
    A_NSHORT,                    // contacts shorter than dur_min
    A_DMAX,                      // sum of max overlap / (Ri+Rj)
    A_NANTI,                     // collisions that gained energy
    A_DEFRAC,                    // sum dE/E_pre
    A_DETR,                      // sum of change of relative trans. energy
    A_DEROT,                     // sum of change of rotational energy
    A_NSAME,                     // collisions with the previous partner
    A_NTIP,                      // collisions touching a rod end cap
    A_UU,                        // sum |ui.uj| at impact
    A_NMULTI,                    // collisions with >2 bodies involved
    A_NN,                        // 6: sum n_a n_b at impact (fabric)
    A_FFN = A_NN + 6,            // free flights: count, sum, sum^2
    A_FFS,
    A_FFS2,
    NACC
  };
  enum { M_DURMIN = 0, M_NEGDURMAX, M_NEGDMAXMAX, NMIN };    // MPI_MIN slots (max stored as -max)

  double acc[NACC];
  double accmin[NMIN];
  double uel_step;     // elastic energy at current step (owned contacts)
  double G[6];         // velocity gradient h_rate*h_inv (Voigt), current step
  double dur_min_thresh;
  double dur_long_thresh;    // 5x the median duration of the previous window (steps)
  // contact-duration histogram: bin k covers [10^(k/20), 10^((k+1)/20)) steps
  static constexpr int NDH = 140;
  double durhist[NDH];       // local, current window
  void add_duration(double steps);

  // called by the pair style
  void pair_begin();
  void contact_start_atoms(int i, int j, int nlocal, double tnow);
  void contact_end_atoms(int i, int j, int nlocal, double tnow);
  bool sample_pair(tagint a, tagint b) const;
  void push_event(const double *rec);

  static constexpr int NEV = 22;    // doubles per event record
  double **pa;                      // per-atom data

 private:
  int nwindow, nsample, appendflag;
  double log_fraction;
  char *prefix;
  FILE *fp_stress, *fp_energy, *fp_coll, *fp_struct, *fp_sensor, *fp_events;
  class PairGranSpherocylHistory *pair;
  double *Rtype, *Htype;

  // time bookkeeping
  double t_win, strain_total, t_run, t_since_rot;
  bigint nsteps_win;
  double e_prev;           // total mechanical energy at the start of the window
  double wshk_rate_prev;   // local kinetic shear-work rate at the previous step
  double pc0[3];           // peculiar momentum at setup
  bool e_prev_valid;

  // structure accumulators (global, identical on all procs)
  static constexpr int NMODE = 10, NJX = 3, NEM = 2, NGRID = 3;
  enum {
    S_SK = 0,
    S_JX = S_SK + NMODE,
    S_EM = S_JX + NJX,
    S_DISP = S_EM + NEM,
    S_Q = S_DISP + NGRID,
    NSACC = S_Q + 6
  };
  double sacc[NSACC];
  int nsamp;

  // run-cumulative copies of reduced window sums (for input-script access)
  double racc[NACC];
  double rdurhist[NDH];    // run-cumulative duration histogram (global)
  double raccmin[NMIN];
  double rtime;

  // last-window results exported through compute_vector()
  static constexpr int NVEC = 25;
  double vec[NVEC];

  // event buffer (local)
  std::vector<double> evbuf;
  int warn_count[16];

  void reset_window();
  void finish_window();
  void structure_sample();
  void rotation_sample(double w);
  void instantaneous(double &etr, double &erot, double *pc, double &cmax2, double &c2mean);
  void atom_omega(int i, double *omega, double *inertia);
  void stream_velocity(const double *x, double *u);
  void open_files();
  void warn(int which, const char *msg);
  static double hist_quantile(const double *h, double q);
};

}    // namespace LAMMPS_NS

#endif
#endif
