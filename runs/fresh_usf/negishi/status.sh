#!/bin/bash
# Progress of all cases: stage reached, finished production blocks, sensor flags.
FRESH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FRESH"
ndone=0; ntot=0
for c in $(cat cases.txt); do
  ntot=$((ntot+1))
  if [ ! -f $c/log.lammps ]; then printf "%-12s not started\n" $c; continue; fi
  if grep -q "^DONE" $c/log.lammps; then ndone=$((ndone+1)); st="DONE";
  else st=$(grep -E "^STAGE|^PRODUCTION BLOCK [0-9]+/[0-9]+ done" $c/log.lammps | tail -1 | cut -c1-40); fi
  flags="-"
  [ -f $c/prod.sensors ] && flags=$(awk '!/^#/ && $4!="-"{for(i=1;i<=length($4);i++) f[substr($4,i,1)]++} END{s=""; for(k in f) s=s k f[k] " "; print (s==""?"clean":s)}' $c/prod.sensors)
  printf "%-12s %-42s prod sensors: %s\n" $c "$st" "$flags"
done
echo "$ndone / $ntot cases done"
