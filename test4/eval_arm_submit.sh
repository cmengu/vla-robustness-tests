#!/bin/bash
# Submit the full evaluation of one trained arm: 10 LIBERO-Plus jobs (Test 3's 700 cases; long
# tasks in quarters so each fits 3 h) + 2 object-shift jobs (Test 2, mid-reach 5 and 10 cm).
# Usage: bash eval_arm_submit.sh A      Logs land in ~ (t4e-*.log, t4s-*.log).
ARM=${1:?ARM A|B|C}
POLICY="${WORK:-$HOME/test4_work}/arm_$ARM/checkpoints/last/pretrained_model"
[ -f "$POLICY/config.json" ] || { echo "no checkpoint at $POLICY"; exit 1; }
D=$(cd "$(dirname "$0")" && pwd); cd $HOME
G=(h100-47 a100-40 a100-80 h200-141); i=0
for S in libero_spatial libero_object libero_goal; do for H in a b; do
  sbatch -J e$ARM-$S-$H --gpus=${G[i%2]} $D/slurm_eval_arm.sh $ARM $S $H; i=$((i+1))
done; done
for H in a b; do for P in 1 2; do
  sbatch -J e$ARM-libero_10-$H$P --gpus=${G[i%2]} $D/slurm_eval_arm.sh $ARM libero_10 $H $P; i=$((i+1))
done; done
for CM in 5 10; do sbatch -J s$ARM-$CM --gpus=h100-47 $D/slurm_shift_arm.sh $ARM $CM; done
squeue --me -o "%.8i %.18j %.8T %.14r"
