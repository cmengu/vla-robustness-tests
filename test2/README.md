# Test 2: can the policy react to a moved object?

On LIBERO-Object, the object the task names ("pick up the X ...") is slid sideways along
the table partway through an episode. We then check whether the policy goes to the new
spot. The models are not changed. Only the rollout wrapper changes.

`eval_shift.py` is `test1/eval_delay.py` with the shift added. The rollout loop and the
`--griffin.*` delay logic are the same: an AST comparison of `chunk_rollout`, with the
`tracker` lines removed, matches eval_delay exactly. With `--shift.mode=none` the policy
gets the same observations and takes the same actions as under eval_delay.

## Conditions

| flag | meaning |
|---|---|
| `--shift.mode=none` | no shift (baseline); still writes the per-episode log |
| `--shift.mode=step0` | shift right after reset, before the first action |
| `--shift.mode=midreach` | shift on the first step the gripper is within `radius` of the object |
| `--shift.mode=blind` | like midreach, but the policy keeps getting the camera frames from just before the shift. The robot state stays live |
| `--shift.size=0.05` | shift distance in metres |
| `--shift.radius=0.15` | trigger distance: 3D, from the gripper's grip site to the object's body origin |
| `--shift.seed=0` | seed for the shift direction |
| `--griffin.delay`, `--griffin.n_action_steps`, `--griffin.fm_num_steps` | same as Test 1 |

How the shift works:
- **Target object.** The first argument of the BDDL goal `(In <object> basket_1_contain_region)`.
  If that is missing, it falls back to the first `obj_of_interest`.
- **Direction.** Drawn uniformly in the table plane from `rng(seed, task_id, start_state)`.
  Every mode, size and model gets the same sequence of directions for the same episode.
- **Rejection.** A direction is skipped if the new spot is within 0.06 m of another
  object's centre (0.13 m for the basket), or if it leaves the box around all objects at
  the start of the episode, enlarged by 0.05 m on each side. Up to 20 directions are tried.
  If none fit, the episode is logged as `no_valid_shift` and nothing is moved.
- **The move.** The object's free joint is moved in x and y. Its z and orientation stay
  the same, its velocity is set to zero, and then `sim.forward()` runs. In step0 and
  midreach mode the cameras are re-rendered at once, so the next policy input already
  shows the moved object.
- **Batches.** Each env in a batch triggers on its own.

The simulator is reached through the vector env's sub-envs: `env.envs` in LeRobot 0.6.1
and `env._envs` in LeRobot main. Those sub-envs live in the main process only when
`--eval.use_async_envs=false`. The script adds that flag if you leave it out. The shift
modes stop with an error if they cannot reach the simulator. The none mode logs without
it.

## Output: `<output_dir>/shift_episodes.jsonl`

The file sits next to `eval_info.json`. It has one line per episode, appended after each
batch.

| field | meaning |
|---|---|
| `condition`, `size`, `radius`, `seed`, `delay`, `n_action_steps` | run settings |
| `task_id`, `init_state` | LIBERO task (0-9) and start-state index |
| `target` | the object that was moved, e.g. `milk_1` |
| `status` | `shifted`, `no_trigger` (gripper never came within the radius), `no_valid_shift`, or `none` |
| `shift_step` | env steps taken before the shift (0 for step0); null if no shift |
| `reach_step` | first step the gripper was within the radius of the original spot, logged in every mode. In midreach it equals `shift_step` |
| `old_xy`, `new_xy` | object position (world frame, metres) before and after the shift |
| `dir_index` | which of the seeded directions was used (0 means the first one fit) |
| `success`, `steps` | task success and steps until done (max 280) |
| `turn_time` | steps after the shift until the gripper starts 5 steps in a row, each closer to the new spot than to the old one and closer to the new spot than the step before. Null if that never happens |

Quick summary on the cluster:

```
python - <<'EOF'
import json, glob, collections
rows = [json.loads(l) for f in glob.glob('t2-out/*/shift_episodes.jsonl') for l in open(f)]
g = collections.defaultdict(list)
for r in rows: g[(r['condition'], r['size'])].append(r)
for k, v in sorted(g.items()):
    tt = [r['turn_time'] for r in v if r['turn_time'] is not None]
    print(k, len(v), 'success', sum(r['success'] for r in v),
          'shifted', sum(r['status'] == 'shifted' for r in v), 'turned', len(tt))
EOF
```

## The pilot

There are 20 episodes per job: 10 tasks with 2 start states each (`--eval.n_episodes=2`
is per task). For each model, `none` runs once, because size does not apply to it. Then
sizes 2, 5 and 10 cm run in modes step0, midreach and blind. That is 10 jobs per model
and 20 jobs in all.

The batch size is 2 for both models. LeRobot caps the batch size at n_episodes, so
Alpha-S cannot use 4 here. Fast-WAM already ran with 2.

## Putting it on the cluster

Paste each file into the terminal as a `cat > FILE <<'EOF' ... EOF` block. Every line is
100 characters or fewer, and no file contains a line that reads just `EOF`.

1. `mkdir -p ~/t2`
2. `cat > ~/alpha-s/examples/libero/eval_shift.py <<'EOF'`, paste `eval_shift.py`, then `EOF`
3. `cat > ~/t2/pilot.sh <<'EOF'`, paste `pilot.sh`, then `EOF`
4. `cat > ~/t2/pilot-submit.sh <<'EOF'`, paste `pilot-submit.sh`, then `EOF`
5. Check the files: `md5sum ~/alpha-s/examples/libero/eval_shift.py ~/t2/*.sh`. Compare
   the output against the md5s in the report.
6. Run one job first:
   `sbatch --gpus=h100-47 --exclude=xgpi15,xgpj0 -o ~/t2-smoke-%j.log ~/t2/pilot.sh alphas 5 midreach`
   - In the log, look for `EPISODES LOGGED: 20`.
   - In the JSONL, `target` should match the task name and `status` should be mostly
     `shifted`.
   - `shift_step` should be well before `steps`.
   - Save a video from `videos/` and confirm that the object really slides.
7. If the smoke job looks right, run `bash ~/t2/pilot-submit.sh`.
