#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100-47
#SBATCH --exclude=xgpi15
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=$HOME/t3pa-%j.log
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/osmesa/lib
export LIBERO_CONFIG_PATH=$HOME/.libero-plus PYTHONPATH=$HOME/libero-plus
. $HOME/aplus-venv/bin/activate
cd $HOME/alpha-s
t0=$(date +%s)
python examples/libero/eval_delay.py --policy.path=griffinlabs/Griffin-Alpha-S-LIBERO --policy.device=cuda --env.type=libero_plus --env.task=libero_10 --env.task_ids="[18,199,447,721,1053,1313,1615,1952,2164,2416]" --env.observation_height=256 --env.observation_width=256 --env.max_parallel_tasks=1 --eval.n_episodes=1 --eval.batch_size=1 --output_dir=outputs/t3/pilot-alphas
echo "TOOK $(( $(date +%s) - t0 )) s"
