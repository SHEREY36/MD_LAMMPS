#!/bin/bash
# Progress of the fresh HCS sweep: finished / running / not started.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
done=0; run=0; todo=0
while read c; do
  if grep -q "^DONE" "$c/log.lammps" 2>/dev/null; then done=$((done+1))
  elif [ -f "$c/log.lammps" ]; then run=$((run+1)); else todo=$((todo+1)); fi
done < cases.txt
echo "finished $done   running/stopped $run   not started $todo   (of $(grep -c . cases.txt))"
