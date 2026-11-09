# VLA robustness tests

Does training a robot policy as a world-action model make it more robust? Four stress tests on
two open-weight LIBERO policies, run in simulation on one GPU each:

- **Alpha-S**: [Griffin-Alpha-S-LIBERO](https://huggingface.co/griffinlabs/Griffin-Alpha-S-LIBERO),
  a plain vision-language-action model (Qwen3-VL 4B backbone, flow-matching action expert).
- **Fast-WAM**: [fastwam_libero_uncond_2cam224](https://huggingface.co/lerobot/fastwam_libero_uncond_2cam224),
  trained with a world-action objective ([arXiv 2603.16666](https://arxiv.org/abs/2603.16666)).

**Result so far: the world-action model is no more robust, and under changed scenes it is 29 points
worse** (51% vs 80% on 613 paired LIBERO-Plus cases). Full tables: [RESULTS.md](RESULTS.md).

| Test | Question | Folder | Status |
|---|---|---|---|
| 1 | Does success hold when actions arrive late? | [test1](test1) | done |
| 2 | Can the policy follow an object that moves mid-episode? | [test2](test2) | done |
| 3 | Does success hold when the scene is perturbed (LIBERO-Plus)? | [test3](test3) | Alpha-S: 87 of 700 cases still running |
| 4 | Does adding a 1-second future-prediction head to Alpha-S help? | [test4](test4) | training |

Tests 1-3 compare two different models; only Test 4 isolates the world-model objective.

## How the tests work

- `test1/eval_delay.py`: LeRobot's LIBERO eval with a delay buffer. An action computed from the frame
  at step t is applied at step t + d. `--griffin.delay=d`; `--griffin.n_action_steps` sets how often the
  policy re-plans (1 = every step, 10 = every chunk). `eval_delay.patch` is the diff against the stock script.
- `test2/eval_shift.py`: the same loop with the target object slid 2, 5 or 10 cm sideways, either before
  the first action (`step0`) or when the gripper first comes within 15 cm (`midreach`). See `test2/README.md`.
- `test3/`: the stock eval over a fixed sample of 700 [LIBERO-Plus](https://arxiv.org/abs/2510.13626) cases
  (`t3_selection.json`), split into Slurm jobs that fit a 3-hour limit.
- `test4/`: fine-tunes Alpha-S with LeRobot's own training loop plus an optional head that predicts the
  V-JEPA 2 embedding of the main camera 1 s ahead. Nothing in LeRobot or Griffin's plugin is edited;
  see `test4/README.md` for the design and the unit tests.

## Running

Each folder's `*.sh` is a Slurm job as run on a university cluster (one H100/H200/A100 per job).
Paths are set in each script's header or in `test4/env.sh`. Environments: LeRobot 0.6.1 with Griffin's
`alpha-s` plugin for Alpha-S; LeRobot 0.6.1 for Fast-WAM; LIBERO-Plus from its own repository for Test 3.
Results in RESULTS.md come from the `eval_info.json` each job writes.
