#!/bin/bash
# Fast-WAM Test 3: 6 halves + libero_10 in quarters = 10 jobs, one at a time on h200-141.
for S in libero_spatial libero_object libero_goal; do
  for H in a b; do sbatch -J f-$S-$H ~/t3f.sh $S $H; done
done
for H in a b; do
  for P in 1 2; do sbatch -J f-libero_10-$H$P ~/t3f.sh libero_10 $H $P; done
done
squeue -u $USER -o "%i %j %T"
