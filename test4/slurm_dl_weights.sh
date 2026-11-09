#!/bin/bash
#SBATCH -J t4-dl-weights
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o %x-%j.out
# CPU job: download the Griffin checkpoint, V-JEPA 2 weights and the Qwen3-VL processor files,
# then run the two CPU test files in the Alpha-S venv (no downloads, ~2 minutes).
source "$HOME/test4/env.sh"
hf download griffinlabs/Griffin-Alpha-S-LIBERO --local-dir "$CKPT"
hf download facebook/vjepa2-vitl-fpc64-256 --local-dir "$VJEPA"
# processor only (tokenizer + configs), into the default HF cache that the input step reads
hf download Qwen/Qwen3-VL-4B-Instruct --include "*.json" --include "*.txt" --include "*.jinja"
python -c "import transformers as t; from transformers import VJEPA2Model; print(t.__version__)"
python -c "import lerobot, lerobot_policy_griffin_alpha; print('lerobot', lerobot.__version__)"
python test_wam.py
python test_real_tiny.py
du -sh "$CKPT" "$VJEPA"
