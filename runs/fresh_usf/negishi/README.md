# Running the fresh USF production on Negishi (Purdue RCAC)

Negishi: 450 standard nodes × 128 cores (AMD EPYC 7763), 256 GB (2 GB per core),
HDR100 InfiniBand. Jobs share nodes, so one array task per case, each with 4–8
ranks, packs efficiently and backfills quickly.

There are two ways to run:
- **Your account** (`-A morri353`, normal QOS): priority on the cores your group bought.
- **`standby`** (`-q standby`): idle cores for free, but each job is limited to **4 h**.

Cases are small (1,250–5,000 particles in a 64 d box). Four ranks is the efficient choice; 8 ranks only helps the largest or longest cases.

## 1. Get the code (once)

```bash
cd /scratch/negishi/$USER                 # run from scratch, not $HOME
git clone https://github.com/SHEREY36/MD_LAMMPS.git && cd MD_LAMMPS
bash runs/fresh_usf/negishi/build_lammps.sh
```

`build_lammps.sh` does four things:
1. Loads the modules in `modules.sh`.
2. Installs the GRANULAR and ASPHERE packages, which copies the spherocylinder code from `src/GRANULAR` into `src/`.
3. Builds `lmp_mpi` and copies it to `runs/fresh_usf/bin/lmp_mpi` (the frozen binary every job uses).
4. Runs a 10-second virial check: the output must show `Pxy=0.001185854123`.

If `module load gcc openmpi` doesn't match Negishi's module names, edit `modules.sh` once (`module avail openmpi`). Both the build and the jobs read it.

Optional but recommended: run the regression suite in an interactive job before production.
```bash
sinteractive -A morri353 -p cpu -n 4 -t 2:00:00
cd runs/regression/v2 && LMP=../../fresh_usf/bin/lmp_mpi ./run_regression.sh && python3 check_regression.py
```
Expect 33/33 PASS. T6 is a sphere run through the production template, compared against DSMC.

## 2. Generate the cases and job lists

```bash
module load anaconda            # any python3 with numpy
cd runs/fresh_usf/negishi
python3 prepare_cases.py        # all 50 cases (AR 2, 1.5, 2.5, 3, 1 × α 0.50–0.95)
```

The data files use fixed seeds, so they are identical to the local ones. `jobs/cost_table.txt` lists predicted steps, ranks, wall time and core-hours per case. With the defaults (`--max-wall 4 --cores 256`) the whole set is about 560 core-hours and needs 236 cores at once. Every case is predicted to finish in 4 h or less (long cases get 8 or 16 ranks), and time limits are 2× the prediction.

The cases are grouped into a few arrays (e.g. `jobs/np4_t4h.txt`, `jobs/np4_t8h.txt`, …). Useful options:
- `--standby`: arrays with a time limit ≤ 4 h go to the free standby QOS (not needed while the account has free cores).
- `--max-wall H`: target predicted hours per case (default 4); cases above it get 8 or 16 ranks.
- `--ARs 2 --alphas 0.5 0.55`: generate only a subset.

## 3. Submit and monitor

```bash
bash jobs/submit_all.sh          # one sbatch per array
squeue -u $USER
bash status.sh                   # stage / blocks done / production sensor flags per case
```

Each array task runs one case directory (`../AR*/a*/`), skips cases already DONE, and logs to `logs/`. A killed or timed-out case restarts from the beginning on resubmission. Time limits are 2× the prediction, so timeouts should be rare.

Check the first finished jobs: compare `Loop time` in `log.lammps` with `jobs/cost_table.txt`. If Negishi is faster or slower than predicted, rerun `prepare_cases.py --estimate-only --rate <atom-steps/s/rank>` before submitting the rest.

## 4. Analyse

```bash
cd runs/fresh_usf && python3 postprocess_fresh_usf.py     # summary.csv + diagnostics_report.txt
```

Copy back only what you need, e.g. `rsync -av --include='*/' --include='prod.*' --include='prof_*.dat' --include='rdf.dat' --include='params.in' --include='log.lammps' --exclude='*' negishi:.../runs/fresh_usf/ ./`. Snapshots and restart files are large.
