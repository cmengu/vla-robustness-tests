# Test 4: a small world-action model from Griffin Alpha-S

## What we are testing

Griffin Alpha-S plans robot actions from camera images. We give it one extra "future lesson"
while it fine-tunes on LIBERO: a small head reads the backbone's picture of the main camera and
has to predict how a frozen video model (V-JEPA 2) will see that camera 1 second later.

    loss = flow loss (the normal action loss) + w * future loss        (w = 0.1 by default)

At evaluation time the head is thrown away. The saved checkpoint is a normal Griffin checkpoint,
so the stock LIBERO evaluation loads it unchanged.

Three arms, all starting from griffinlabs/Griffin-Alpha-S-LIBERO, all batch 16 x 6,000 steps:

- A: the stock recipe, no head. Same extra training, so it is the fair baseline.
- B: + future head. Target = change in the V-JEPA vector: emb(t+10) - emb(t).
- C: + the same head. Target = the current V-JEPA vector emb(t). This checks whether
  "any extra head helps" rather than "predicting the future helps".

## Facts checked in the sources

- Recipe (alpha-s README.md and docs/finetuning.md): `lerobot-train --policy.path=
  griffinlabs/Griffin-Alpha-S-LIBERO --dataset.repo_id=HuggingFaceVLA/libero --batch_size=16
  --steps=6000`. Learning rates 3e-5 (backbone) and 5e-5 (expert), cosine schedule.
- Dataset HuggingFaceVLA/libero (meta/info.json): 1,693 episodes, 273,465 frames, 10 fps,
  377 parquet files (~35 GB). Images are PNGs stored inside the parquet files, 256 x 256.
- Main camera key: `observation.images.image` (agent view, upright); wrist: `...image2`.
- Prompt: about 200 tokens per sample, 64 of them for the main camera image (measured on the
  real Qwen3-VL processor with a tiny model).
- V-JEPA 2 is part of transformers 5.5 (`VJEPA2Model`), so **no extra pip install is needed**
  in ~/griffin-venv. pyarrow and Pillow come with lerobot.

## Files

| file | what it does |
|---|---|
| `env.sh` | paths used by every job (edit once). Activates ~/griffin-venv, sets TMPDIR. |
| `vjepa_targets.py` | runs V-JEPA 2 once over every frame; one 1,024-number vector per frame |
| `wam_head.py` | head, hidden-state hook, main-camera pooling, target lookup, mask, loss |
| `train_wam.py` | training entry point: `--arm A/B/C`; runs lerobot's own training loop |
| `probe.py` | after training: did the future lesson reach the backbone? (B vs A vs C) |
| `test_wam.py` | 25 unit tests with tiny fake modules (torch only) |
| `test_real_tiny.py` | 6 tests on the REAL Griffin policy, tiny random backbone, no downloads |
| `slurm_*.sh` | the cluster jobs, in run order below |

### How the pieces fit (no package is edited)

`train_wam.py` imports lerobot's `lerobot_train` module, swaps four of its functions for thin
wrappers, then calls lerobot's `train()`:

1. `make_policy`: builds the stock policy, then gives it our forward (arm A: stock forward plus
   two logging numbers; B/C: also the head).
2. `make_pre_post_processors`: wraps the preprocessor so each batch keeps `episode_index` and
   `frame_index` (Griffin's input step would drop them; we need them to look up the target).
3. `save_checkpoint`: the stock save, with the head removed from `model.safetensors` (checked
   on every save). The head goes to `wam_head.safetensors` next to `pretrained_model/` only so
   training can resume. Older checkpoints are deleted (`--keep_last 2`; each is ~26 GB).
4. `update_policy`: the stock step, then prints `WAM_STATS` lines (peak GPU memory, tokens per
   sample, seconds per step).

Which hidden state the head reads, and why: the backbone has 36 layers. The action expert does
not read the backbone's final output; at every layer it reads that layer's keys and values,
which are computed from the layer's INPUT. So the last layer's own output, and the final norm
after it, feed nothing the expert uses. If the head read the final output, the backbone could
learn the future lesson only in that unused last step, and the actions would never benefit.
So we hook the output of layer 35 of 36 (`language_model.layers[-2]`), which is exactly the
input of the last layer. The lesson then has to live where the last layer's keys and values
(and so the expert) can see it. Layer choice: `--hook_layer` (default -2), saved in
`wam_config.json`.

This tensor is the raw residual stream, before any normalisation, and Qwen has a few very large
outlier numbers in it. So the head first applies a LayerNorm with no learned weights (nothing new
is saved in the policy). Qwen3-VL's extra visual injection ("deepstack") only touches the first
3 layers (transformers 5.5.4, modeling_qwen3_vl.py:915-932), so layer 35's output is plain.

The hook only reads the tensor and is removed after every forward, so the flow loss is
computed exactly as in the stock code. We average the tensor over the main camera's 64 image
tokens (the first run of image tokens in the prompt), giving one 2,560-number vector per sample.
The head is norm -> 2560 -> 2048 -> 1024 (7.3M parameters), its last layer starting at zero.

Targets are standardised per number (mean 0, std 1 over the dataset) so the future loss starts
near 1.0 in both B and C, and w means the same thing in both. Frames whose t+10 is past the
episode end are left out of the future loss (about 6% of frames); C uses the same mask as B.

## Run order

Copy all files to `~/test4`. Every command is run from `~/test4`.

| # | job | partition | time | what to look for |
|---|---|---|---|---|
| 1 | `sbatch slurm_dl_data.sh` | normal | ~1-8 h | "parquet files 377", "frames 273465" |
| 2 | `sbatch slurm_dl_weights.sh` | normal | <1 h | "25/25 passed", "6/6 passed" |
| 3 | `sbatch slurm_targets.sh` | gpu | <1 h | "parts done: 1693 / 1693", then the merge line |
| 4 | `sbatch slurm_pilot.sh` | gpu | <3 h | `WAM_STATS` lines (see below) |
| 5 | `sbatch slurm_arm.sh A` (and B, C) | gpu-long | per pilot | final `WAM_FINAL` line |
| 6 | `sbatch slurm_probe.sh` | gpu | ~1-2 h | the table printed at the end |

Jobs 1 and 2 can run together. Job 3 needs 1 and 2. Job 4 needs 3.

The three arms can run at the same time, one per GPU type (one of each per user):

    sbatch slurm_arm.sh A
    sbatch --gres=gpu:a100-80:1 --exclude=xgpj0 slurm_arm.sh B
    sbatch --gres=gpu:h200-141:1 slurm_arm.sh C

If an arm hits its time limit, `sbatch` the same line again: it resumes from its last checkpoint.
If a run dies before its first checkpoint, delete `~/test4_work/arm_X` before resubmitting
(lerobot refuses to write into an existing folder).

To try another weight: `sbatch --export=ALL,W=0.03 slurm_pilot.sh` (same for slurm_arm.sh; a
different W needs a fresh output folder, so rename `arm_B` first).

## What the pilot must tell us

Read the `WAM_STATS` lines and lerobot's `step:` lines in `t4-pilot-<jobid>.out`:

1. **Memory**: `peak_gpu_mem_gb` against `gpu_total_gb`. Around 43 GB means bf16 weights and
   optimizer; around 86 GB means fp32 master weights and it will NOT fit an a100-80.
2. **Prompt length**: `tokens_per_sample_mean` should be ~200 and `main_image_tokens_mean` 64.
3. **Speed**: `sec_per_step` x 6,000 = the time each arm needs. Set `--time` in slurm_arm.sh to
   that plus 20%.
4. **The weight w**: `future_loss` starts near 1.0 and should fall. Compare `flow_loss` with
   0.1 x `future_loss`: if the second is much bigger than the first, the future lesson dominates;
   try W=0.03 or 0.01. `future_valid` should be about 0.94.
5. **Attention**: the log line "Attention implementation:" should say flash_attention_2. If it
   says sdpa, flash-attn is missing and every arm is slower (same for all three, so still fair).

Note: a 500-step run squeezes lerobot's learning-rate warmup and decay into 500 steps, so its
loss curve is not the start of the 6,000-step curve.

## Outputs

- `~/test4_work/targets/vjepa_targets.npy`: 273,465 x 1,024 fp16 (~560 MB), and
  `vjepa_index.npz` (episode and frame of each row, plus the settings used).
- `~/test4_work/arm_X/checkpoints/last/pretrained_model/`: a plain griffin_alpha checkpoint.
  Evaluate it with the stock LIBERO eval:
  `POLICY_PATH=<that folder> bash ~/alpha-s/examples/libero/run_eval.sh`.
- `~/test4_work/probe/*.json`: held-out R^2 and cosine for each checkpoint and layer.

## V-JEPA target settings (vjepa_targets.py)

For each frame t: a clip of 8 frames ending at t, every 2nd frame (t-14, t-12, ..., t; frames
before the episode start repeat frame 0). That is 1.4 s of video at an effective 5 fps. Frames
are resized to 292 and centre-cropped to 256 with ImageNet colours, as V-JEPA 2's own processor
does (checked against it: max difference 0.009 after normalisation, from rounding). The encoder's
1,024 tokens are averaged into one 1,024-number vector, stored as fp16. About 0.7 TFLOP per clip,
~2e17 FLOP in total: well under an hour on one A100/H100.

## The probe (probe.py)

It freezes a checkpoint, takes 4,000 frames from 80% of the episodes and 1,000 frames from the
other 20% (same frames for every checkpoint, set by `--seed`), and fits a ridge regression from
the pooled hidden state to the standardised change target. It reports held-out R^2 and cosine.
It also probes the current target as a reference. The job runs it at `--layer -2` (the input
of the last layer: what the head saw, and what the expert reads through the last layer's keys
and values) and at `--layer 17` (halfway down), to see how deep the lesson went.

## Local checks already done (on a Mac, CPU)

- `python3 -I test_wam.py`: 25/25 pass. `test_real_tiny.py` (real Griffin policy, tiny random
  3-layer Qwen3-VL, transformers 5.5.4, lerobot 0.6.1): 6/6 pass. They check that the future
  loss's gradient reaches layers[-2] and does NOT reach the last layer's MLP.
- An end-to-end run of `train_wam.py` with a tiny random Griffin checkpoint, the real Qwen3-VL
  processor and 3 real LIBERO episodes: arms A, B, C train, save, resume and prune.
- Arm A vs stock `lerobot-train`, same settings, 2 steps: the saved weights are bit-for-bit equal.
- The saved checkpoints load with `GriffinAlphaPolicy.from_pretrained(..., strict=True)`.
- `vjepa_targets.py` on one real parquet file (3 episodes) with a tiny random V-JEPA.
