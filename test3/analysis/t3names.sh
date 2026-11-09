#!/bin/bash
#SBATCH -p normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=t3names-%j.log
export LIBERO_CONFIG_PATH=$HOME/.libero-plus TMPDIR=$HOME/tmp
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/osmesa/lib
. $HOME/fplus-venv/bin/activate
python -I $HOME/t3names.py
