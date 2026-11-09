#!/bin/bash
#SBATCH -J t4-probe
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -o %x-%j.out
# Probe the untouched checkpoint and arms A, B, C on the same 5,000 frames, at layer -2 (the
# training hook = input of the last of 36 layers) and at layer 17 (halfway down).
source "$HOME/test4/env.sh"
export HF_HUB_OFFLINE=1
mkdir -p "$WORK/probe"
for NAME in base A B C; do
  C_DIR="$WORK/arm_$NAME/checkpoints/last/pretrained_model"
  if [ "$NAME" = base ]; then C_DIR="$CKPT"; fi
  [ -d "$C_DIR" ] || { echo "skip $NAME (no $C_DIR)"; continue; }
  for LAYER in -2 17; do
    python probe.py --checkpoint "$C_DIR" --targets_dir "$TARGETS" --dataset_root "$DATA_ROOT" \
      --layer "$LAYER" --seed 0 --out "$WORK/probe/${NAME}_layer${LAYER}.json"
  done
done
python - <<'PY'
import glob, json, os
for f in sorted(glob.glob(os.path.join(os.environ["WORK"], "probe", "*.json"))):
    r = json.load(open(f))
    c, q = r["change"], r["current"]
    print(f"{os.path.basename(f):22s} change R2 {c['r2_pooled']:.3f} cos {c['cosine_mean']:.3f}"
          f" | current R2 {q['r2_pooled']:.3f}")
PY
