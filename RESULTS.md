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
cases for both models. 656 cases had both models at the time of analysis (Alpha-S's last 44
long-task cases were still running; Fast-WAM had all 700, overall 51.4%).

### By suite

| Suite | Cases | Alpha-S | Fast-WAM | Gap |
|---|---:|---:|---:|---:|
| Spatial | 175 | 85.7 | 51.4 | 34.3 |
| Object | 175 | 88.0 | 71.4 | 16.6 |
| Goal | 175 | 70.3 | 40.0 | 30.3 |
| Long tasks | 131 | 77.1 | 42.7 | 34.4 |
| All paired | 656 | 80.5 | 52.0 | 28.5 |

### By perturbation factor

"A>F" = cases Alpha-S passed and Fast-WAM failed; "F>A" the reverse; p = exact two-sided McNemar.

| Factor | n | Alpha-S | Fast-WAM | Gap | A>F | F>A | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Camera viewpoint | 100 | 75.0 | 25.0 | 50.0 | 54 | 4 | <0.0001 |
| Robot start pose | 100 | 76.0 | 41.0 | 35.0 | 39 | 4 | <0.0001 |
| Table layout | 100 | 89.0 | 55.0 | 34.0 | 38 | 4 | <0.0001 |
| Sensor noise | 75 | 68.0 | 38.7 | 29.3 | 29 | 7 | 0.0003 |
| Language rewording | 88 | 86.4 | 65.9 | 20.5 | 21 | 3 | 0.0003 |
| Extra objects | 93 | 78.5 | 58.1 | 20.4 | 21 | 2 | 0.0001 |
| Lighting | 100 | 88.0 | 79.0 | 9.0 | 13 | 4 | 0.049 |

Per-suite-by-factor rows (25 cases each): [test3/analysis/breakdown-2026-11-09.txt](test3/analysis/breakdown-2026-11-09.txt).
Largest single cells: Spatial / robot start pose 92 vs 20; Spatial, Object / camera viewpoint 84 vs 28
and 88 vs 32; Long / table layout 84 vs 28. Cells where the models tie: Object / language 92 vs 92;
Object / extra objects 76 vs 80; Object and Spatial / lighting 100 vs 96 and 88 vs 84.

Harness check: Fast-WAM on 10 unperturbed LIBERO cases through the same code path: 100.

## Test 4: future-prediction head on Alpha-S

Pilot, arm B, 500 steps, batch 16, H100 96 GB: peak GPU memory 49.1 GB; 197 prompt tokens per
sample, 64 for the main camera; 1.62–1.65 s per step; flow loss 0.064–0.068; future loss 0.95–0.97
(from 1.0); future cosine 0.20; fraction of frames with a valid 1-s-ahead target 0.91–0.95.
Arms A, B, C: training, one at a time.
