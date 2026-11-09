#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h200-71
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=$HOME/t3pf-%j.log
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$HOME/osmesa/lib
export LIBERO_CONFIG_PATH=$HOME/.libero-plus PYTHONPATH=$HOME/libero-plus
. $HOME/fplus-venv/bin/activate
cd $HOME/lerobot
t0=$(date +%s)
python $HOME/alpha-s/examples/libero/eval_delay.py --policy.path=lerobot/fastwam_libero_uncond_2cam224 --env.type=libero_plus --env.task=libero_10 --env.task_ids="[18,199,447,721,1053,1313,1615,1952,2164,2416]" --env.episode_length=520 --env.observation_height=224 --env.observation_width=224 --env.max_parallel_tasks=1 --eval.n_episodes=1 --eval.batch_size=1 --policy.device=cuda --policy.dtype=float32 --policy.n_action_steps=10 --output_dir=outputs/t3/pilot-fastwam
echo "TOOK $(( $(date +%s) - t0 )) s"
nvidia-smi --query-gpu=name,memory.total --format=csv
