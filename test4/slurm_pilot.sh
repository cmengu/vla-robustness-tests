#!/bin/bash
#SBATCH -J t4-pilot
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH -o %x-%j.out
# 500-step pilot of arm B (the heaviest arm). Saves nothing (home quota); stats only. Checks: peak GPU memory, tokens per sample,
# seconds per step, and whether w=0.1 is sensible (compare flow_loss with 0.1*future_loss).
# Other GPU: sbatch --gres=gpu:a100-80:1 --exclude=xgpj0 slurm_pilot.sh
# Other weight: sbatch --export=ALL,W=0.03 slurm_pilot.sh
source "$HOME/test4/env.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT="$WORK/pilot_B_w${W:-0.1}_${SLURM_JOB_ID:-x}"
python train_wam.py --arm B --steps 500 --save_freq 500 --log_freq 25 --keep_last 1 --weights_only --save_checkpoint=false \
  --future_weight "${W:-0.1}" --output_dir "$OUT" --policy_path "$CKPT" \
  --dataset_root "$DATA_ROOT" --targets_dir "$TARGETS"
echo "pilot done: $OUT   (summary: grep WAM_ t4-pilot-${SLURM_JOB_ID:-x}.out)"
