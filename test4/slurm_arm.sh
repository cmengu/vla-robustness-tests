#!/bin/bash
#SBATCH -J t4-arm
#SBATCH -p gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH -o %x-%j.out
# One full arm: 6,000 steps, batch 16 (Griffin's LIBERO recipe). Usage:
#   sbatch slurm_arm.sh A
#   sbatch --gres=gpu:a100-80:1 --exclude=xgpj0 slurm_arm.sh B
#   sbatch --gres=gpu:h200-141:1 slurm_arm.sh C
# Delete $OUT after its eval and probe, before the next arm (each final save ~10 GB).
source "$HOME/test4/env.sh"
export HF_HUB_OFFLINE=1
ARM="${1:?usage: sbatch slurm_arm.sh A|B|C [SEED] [TAG]}"
SEED="${2:-1000}"; TAG="${3:-}"   # e.g. "B 1001 2" -> arm_B2, seed 1001
OUT="$WORK/arm_$ARM$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Weights only (no optimizer state): one ~10 GB save at the final step, so an arm cannot be
# resumed; a crashed arm restarts from scratch. Run arms one at a time (home quota).
python train_wam.py --arm "$ARM" --steps 6000 --batch_size 16 --save_freq 6000 --keep_last 1 \
  --weights_only --seed "$SEED" --future_weight "${W:-0.1}" --output_dir "$OUT" --policy_path "$CKPT" \
  --dataset_root "$DATA_ROOT" --targets_dir "$TARGETS"
echo "final checkpoint: $OUT/checkpoints/last/pretrained_model"
