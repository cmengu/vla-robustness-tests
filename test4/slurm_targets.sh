#!/bin/bash
#SBATCH -J t4-targets
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -o %x-%j.out
# GPU job: V-JEPA 2 embedding of every frame (expected well under 1 h on an A100/H100).
# Resumable: finished episodes are skipped. To split across two jobs:
#   sbatch --export=ALL,EPS="0 847" slurm_targets.sh
#   sbatch --export=ALL,EPS="847 1693" slurm_targets.sh
# The merge step runs automatically once every episode part exists.
source "$HOME/test4/env.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
RANGE=()
if [ -n "${EPS:-}" ]; then RANGE=(--episodes $EPS); fi
python vjepa_targets.py --data_root "$DATA_ROOT" --out_dir "$TARGETS" --model "$VJEPA" \
  --clip_len 8 --stride 2 --pool all --batch_clips 32 --decode_threads 8 \
  ${RANGE[@]+"${RANGE[@]}"}
N=$(ls "$TARGETS/parts" | grep -c '^ep_.*\.npy$' || true)
echo "parts done: $N / 1693"
if [ "$N" -ge 1693 ]; then
  python vjepa_targets.py --data_root "$DATA_ROOT" --out_dir "$TARGETS" --model "$VJEPA" \
    --clip_len 8 --stride 2 --pool all --merge
  ls -la "$TARGETS"
fi
