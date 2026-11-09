#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100-96
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=$HOME/t3fchk-%j.log
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/osmesa/lib
export LIBERO_CONFIG_PATH=$HOME/.libero-plus PYTHONPATH=$HOME/libero-plus
. $HOME/fplus-venv/bin/activate
cd $HOME/lerobot
t0=$(date +%s)
python $HOME/alpha-s/examples/libero/eval_delay.py --policy.path=lerobot/fastwam_libero_uncond_2cam224 --env.type=libero --env.task=libero_10 --env.task_ids="[0,1,2,3,4,5,6,7,8,9]" --env.episode_length=520 --env.observation_height=224 --env.observation_width=224 --env.max_parallel_tasks=1 --eval.n_episodes=1 --eval.batch_size=1 --policy.device=cuda --policy.dtype=float32 --policy.n_action_steps=10 --output_dir=outputs/t3/chk-fastwam
echo "TOOK $(( $(date +%s) - t0 )) s"
nvidia-smi --query-gpu=name,memory.total --format=csv
