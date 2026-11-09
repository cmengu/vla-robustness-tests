#!/bin/bash
#SBATCH -J t4-dl-data
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH -o %x-%j.out
# CPU job: download the LIBERO training dataset (377 parquet files, ~35 GB). Resumable:
# if it times out, just sbatch it again.
source "$HOME/test4/env.sh"
mkdir -p "$DATA_ROOT"
hf download HuggingFaceVLA/libero --repo-type dataset --local-dir "$DATA_ROOT"
python - <<'PY'
import glob, json, os
root = os.environ["DATA_ROOT"]
info = json.load(open(os.path.join(root, "meta", "info.json")))
n = len(glob.glob(os.path.join(root, "data", "chunk-*", "file-*.parquet")))
print("episodes", info["total_episodes"], "frames", info["total_frames"], "fps", info["fps"])
print("parquet files", n, "(expect 377)")
PY
du -sh "$DATA_ROOT"
