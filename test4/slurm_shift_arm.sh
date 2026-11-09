#!/bin/bash
#SBATCH -p gpu
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=t4s-%x-%j.log
# Test 4 evaluation: one trained arm on Test 2's mid-reach object shift (libero_object, 20 episodes).
# Usage: sbatch -J s<ARM>-<CM> slurm_shift_arm.sh ARM CM      CM = 5 or 10 (cm)
# Same settings as test2/pilot.sh (alphas branch, mode midreach); only the policy path differs.
# Output: ~/t2-out/arm<ARM>-midreach-<CM>cm-<jobid>/ (picked up by res.py).
set -eo pipefail
ARM=${1:?ARM A|B|C}; CM=${2:?CM 5|10}
case $CM in 5) S=0.05 ;; 10) S=0.10 ;; *) echo "bad CM '$CM'"; exit 1 ;; esac
WORK="${WORK:-$HOME/test4_work}"
POLICY="$WORK/arm_$ARM/checkpoints/last/pretrained_model"
[ -f "$POLICY/config.json" ] || { echo "no checkpoint at $POLICY"; exit 1; }
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=$HOME/osmesa/lib HF_HUB_OFFLINE=1
mkdir -p $HOME/tmp $HOME/t2-out
EVAL=$HOME/alpha-s/examples/libero/eval_shift.py
OUT=$HOME/t2-out/arm$ARM-midreach-${CM}cm-${SLURM_JOB_ID:-local}
. $HOME/griffin-venv/bin/activate
cd $HOME/alpha-s
echo "ARM $ARM POLICY $POLICY MODE midreach SIZE $S OUT $OUT"
md5sum $EVAL
t0=$(date +%s)
python $EVAL --policy.path=$POLICY --env.observation_height=256 --env.observation_width=256 \
  --policy.device=cuda --env.type=libero --env.task=libero_object \
  --env.max_parallel_tasks=1 --eval.n_episodes=2 --eval.batch_size=2 \
  --eval.use_async_envs=false --output_dir=$OUT --griffin.delay=0 \
  --shift.mode=midreach --shift.size=$S --shift.radius=0.15 --shift.seed=0
echo "TOOK $(( $(date +%s) - t0 )) s"
echo "EPISODES LOGGED: $(wc -l < $OUT/shift_episodes.jsonl)"
grep -o '"pc_success": [0-9.]*' $OUT/eval_info.json | tail -1
