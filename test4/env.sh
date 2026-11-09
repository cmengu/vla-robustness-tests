# Shared settings for every Test 4 job. Edit the paths here once; all slurm_*.sh source it.
# Usage inside a job: source "$HOME/test4/env.sh"
source "$HOME/griffin-venv/bin/activate"   # before set -u: some activate scripts need it
set -euo pipefail
export CODE="${CODE:-$HOME/test4}"                       # where these files live
export WORK="${WORK:-$HOME/test4_work}"                  # outputs (needs ~300 GB free)
export DATA_ROOT="${DATA_ROOT:-$HOME/data/libero}"       # HuggingFaceVLA/libero (~35 GB)
export CKPT="${CKPT:-$HOME/ckpts/Griffin-Alpha-S-LIBERO}" # griffin_alpha checkpoint (main)
export VJEPA="${VJEPA:-$HOME/ckpts/vjepa2-vitl-fpc64-256}" # V-JEPA 2 ViT-L weights
export TARGETS="${TARGETS:-$WORK/targets}"               # vjepa_targets.npy + index
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR" "$WORK"
# The Qwen3-VL processor lives in the default HF cache (~/.cache/huggingface).
cd "$CODE"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} python=$(which python)"
