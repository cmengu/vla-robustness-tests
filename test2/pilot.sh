#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=t2-%x-%j.log
# Test 2 pilot job: one model, one shift size, one mode; libero_object, 2 episodes per task
# (start states 0 and 1 of each of the 10 tasks) = 20 episodes.
# Usage: sbatch --gpus=<type> [--exclude=...] pilot.sh MODEL SIZE_CM MODE [DELAY]
#   MODEL alphas|fastwam   SIZE_CM 2|5|10   MODE none|step0|midreach|blind   DELAY default 0
# pilot-submit.sh picks the GPU per model. Results: ~/t2-out/<model>-<mode>-<size>cm-<jobid>/
set -eo pipefail
M=$1; CM=$2; MODE=$3; D=${4:-0}
case $CM in 2) S=0.02 ;; 5) S=0.05 ;; 10) S=0.10 ;; *) echo "bad SIZE_CM '$CM'"; exit 1 ;; esac
case $MODE in none|step0|midreach|blind) ;; *) echo "bad MODE '$MODE'"; exit 1 ;; esac
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=$HOME/osmesa/lib
mkdir -p $HOME/tmp $HOME/t2-out
EVAL=$HOME/alpha-s/examples/libero/eval_shift.py
OUT=$HOME/t2-out/$M-$MODE-${CM}cm-${SLURM_JOB_ID:-local}
case $M in
  alphas)
    . $HOME/griffin-venv/bin/activate
    cd $HOME/alpha-s
    ARGS=(--policy.path=griffinlabs/Griffin-Alpha-S-LIBERO
      --env.observation_height=256 --env.observation_width=256) ;;
  fastwam)
    . $HOME/fastwam-venv/bin/activate
    cd $HOME/lerobot
    ARGS=(--policy.path=lerobot/fastwam_libero_uncond_2cam224
      --env.observation_height=224 --env.observation_width=224
      --policy.dtype=float32 --policy.n_action_steps=10) ;;
  *) echo "bad MODEL '$M'"; exit 1 ;;
esac
ARGS+=(--policy.device=cuda --env.type=libero --env.task=libero_object
  --env.max_parallel_tasks=1 --eval.n_episodes=2 --eval.batch_size=2
  --eval.use_async_envs=false --output_dir=$OUT --griffin.delay=$D
  --shift.mode=$MODE --shift.size=$S --shift.radius=0.15 --shift.seed=0)
echo "MODEL $M MODE $MODE SIZE $S DELAY $D OUT $OUT"
md5sum $EVAL
nvidia-smi --query-gpu=name,memory.total --format=csv
t0=$(date +%s)
python $EVAL "${ARGS[@]}"
echo "TOOK $(( $(date +%s) - t0 )) s"
echo "EPISODES LOGGED: $(wc -l < $OUT/shift_episodes.jsonl)"
grep -o '"pc_success": [0-9.]*' $OUT/eval_info.json | tail -1
