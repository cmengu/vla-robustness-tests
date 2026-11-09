#!/bin/bash
# Submit the Test 2 pilot: per model, one "none" baseline (size does not apply) plus
# 3 sizes x 3 shift modes = 10 jobs per model, 20 jobs in all. Run from anywhere.
# Expects ~/t2/pilot.sh and ~/alpha-s/examples/libero/eval_shift.py.
mkdir -p $HOME/t2-logs $HOME/t2-out $HOME/tmp
for M in alphas fastwam; do
  if [ $M = alphas ]; then G=(--gpus=h100-47 --exclude=xgpi15,xgpj0)
  else G=(--gpus=h200-141); fi
  L=(-o "$HOME/t2-logs/%x-%j.log")
  sbatch -J t2-$M-none "${G[@]}" "${L[@]}" $HOME/t2/pilot.sh $M 5 none
  for CM in 2 5 10; do
    for MODE in step0 midreach blind; do
      sbatch -J t2-$M-$MODE-$CM "${G[@]}" "${L[@]}" $HOME/t2/pilot.sh $M $CM $MODE
    done
  done
done
squeue -u $USER -o "%i %j %T"
