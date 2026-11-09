# Results

As of 9 November 2026. Success rates in percent. Models: Alpha-S = griffinlabs/Griffin-Alpha-S-LIBERO;
Fast-WAM = lerobot/fastwam_libero_uncond_2cam224. Simulator: LIBERO via LeRobot 0.6.1.

## Test 1: actions arrive late

libero_object, 100 episodes per point (two 50-episode halves). Each action is applied d steps
(0.1 s each) after the frame it was computed from.

| Delay (steps) | 0 | 2 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| Fast-WAM | 94 | 94 | 85 | 80 | 41 |
| Alpha-S, re-plan every step (K=1) | 99 | 93 | 95 | 81 | 50 |
| Alpha-S, re-plan every 10 steps (K=10) | 97 | 96 | 89 | 81 | 34 |

## Test 2: the target object moves

libero_object, 20 episodes per cell (start states 0 and 1 of each of the 10 tasks). No move: both 100%.

| Move | 2 cm | 5 cm | 10 cm |
|---|---:|---:|---:|
| At the start: Alpha-S | 100 | 95 | 70 |
| At the start: Fast-WAM | 95 | 70 | 10 |
| Mid-reach: Alpha-S | 100 | 80 | 55 |
| Mid-reach: Fast-WAM | 100 | 60 | 0 |

The 5 cm mid-reach cell was run twice for each model (Alpha-S 80 and 80; Fast-WAM 60 and 60).
"Blind" mode (camera frozen after the move) scored 0 for both models at 2, 5 and 10 cm and is dropped.

## Test 3: perturbed scenes (LIBERO-Plus)

700 cases sampled across four suites, 100 per perturbation factor, one episode each, identical
cases for both models. All 700 have both models.

### By suite

| Suite | Cases | Alpha-S | Fast-WAM | Gap |
|---|---:|---:|---:|---:|
| Spatial | 175 | 85.7 | 51.4 | 34.3 |
| Object | 175 | 88.0 | 71.4 | 16.6 |
| Goal | 175 | 70.3 | 40.0 | 30.3 |
| Long tasks | 175 | 77.1 | 44.6 | 32.5 |
| All | 700 | 80.3 | 51.9 | 28.4 |

### By perturbation factor

"A>F" = cases Alpha-S passed and Fast-WAM failed; "F>A" the reverse; p = exact two-sided McNemar.

| Factor | n | Alpha-S | Fast-WAM | Gap | A>F | F>A | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Camera viewpoint | 100 | 75.0 | 25.0 | 50.0 | 54 | 4 | <0.0001 |
| Robot start pose | 100 | 76.0 | 41.0 | 35.0 | 39 | 4 | <0.0001 |
| Table layout | 100 | 89.0 | 55.0 | 34.0 | 38 | 4 | <0.0001 |
| Sensor noise | 100 | 69.0 | 39.0 | 30.0 | 39 | 9 | <0.0001 |
| Extra objects | 100 | 78.0 | 56.0 | 22.0 | 25 | 3 | <0.0001 |
| Language rewording | 100 | 87.0 | 68.0 | 19.0 | 23 | 4 | 0.0003 |
| Lighting | 100 | 88.0 | 79.0 | 9.0 | 13 | 4 | 0.049 |

Per-suite-by-factor rows (25 cases each, from the 656-case run; the long-task rows there lack the last 44 cases): [test3/analysis/breakdown-2026-11-09.txt](test3/analysis/breakdown-2026-11-09.txt). Final factor table: [breakdown-2026-11-09-final.txt](test3/analysis/breakdown-2026-11-09-final.txt).
Largest single cells: Spatial / robot start pose 92 vs 20; Spatial, Object / camera viewpoint 84 vs 28
and 88 vs 32; Long / table layout 84 vs 28. Cells where the models tie: Object / language 92 vs 92;
Object / extra objects 76 vs 80; Object and Spatial / lighting 100 vs 96 and 88 vs 84.

Harness check: Fast-WAM on 10 unperturbed LIBERO cases through the same code path: 100.

## Test 4: future-prediction head on Alpha-S

Pilot, arm B, 500 steps, batch 16, H100 96 GB: peak GPU memory 49.1 GB; 197 prompt tokens per
sample, 64 for the main camera; 1.62–1.65 s per step; flow loss 0.064–0.068; future loss 0.95–0.97
(from 1.0); future cosine 0.20; fraction of frames with a valid 1-s-ahead target 0.91–0.95.
Arm A trained (6,000 steps, 2 h on an H100, 48.5 GB peak, 1.08 s/step) and arm B trained (6,000 steps,
1.5 h on an H200, 49.1 GB peak, 0.89 s/step). Arm B's future loss fell from 0.94 at step 1 to 0.58 at
step 6,000; the cosine between predicted and true 1-s-ahead embedding rose from 0.01 to 0.59 (pilot: 0.20
after 500 steps). Both arms are being evaluated on Test 3's 700 cases and Test 2's 5 cm and 10 cm
mid-reach shifts; arm C training. Arm A so far (partial): object 84.1 / 90.8 (halves a/b) vs stock 88.6 / 87.4;
spatial 84.1 / 82.8 vs 89.8 / 81.6; goal-b 71.3 vs 67.8; 10 cm mid-reach 60 vs 55; 5 cm 85 vs 80.
