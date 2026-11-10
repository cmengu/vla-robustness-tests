# Results

As of 10 November 2026. Success rates in percent. Models: Alpha-S = griffinlabs/Griffin-Alpha-S-LIBERO;
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

Three arms fine-tuned from griffinlabs/Griffin-Alpha-S-LIBERO, 6,000 steps, batch 16: A plain;
B plus a head predicting the V-JEPA 2 embedding of the main camera 1 s ahead (weight 0.1); C the
same head predicting the current embedding. Pilot (arm B, 500 steps, H100 96 GB): 49.1 GB peak,
1.65 s/step, future loss 0.95–0.97, cosine 0.20.

Training: A 2 h on an H100 (48.5 GB peak, 1.08 s/step); B 1.5 h on an H200 (49.1 GB, 0.89 s/step);
C 1.5 h on an H200 (49.1 GB). B's future loss 0.94 (step 1) → 0.58 (step 6,000), cosine 0.01 → 0.59.
C's loss → 0.08, cosine → 0.96.

### Success on the same 700 LIBERO-Plus cases and the Test 2 shift cells (20 episodes)

| Arm | Spatial | Object | Goal | Long | All 700 | 5 cm mid-reach | 10 cm mid-reach |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stock Alpha-S | 85.7 | 88.0 | 70.3 | 77.1 | 80.3 | 80 | 55 |
| A | 83.4 | 87.4 | 74.3 | 77.1 | 80.6 | 85 | 60 |
| B | 85.7 | 83.4 | 74.9 | 75.4 | 79.9 | 85 | 65 |
| C | 84.6 | 88.0 | 73.7 | 77.7 | 81.0 | 90 | 70 |

### Paired comparisons by factor (n = 100 each), success %

| Factor | A | B | C | B>A / A>B | p | C>A / A>C | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Camera viewpoint | 73 | 74 | 72 | 6 / 5 | 1.00 | 10 / 11 | 1.00 |
| Robot start pose | 68 | 70 | 72 | 10 / 8 | 0.81 | 9 / 5 | 0.42 |
| Table layout | 92 | 94 | 97 | 5 / 3 | 0.73 | 7 / 2 | 0.18 |
| Sensor noise | 69 | 63 | 68 | 6 / 12 | 0.24 | 5 / 6 | 1.00 |
| Extra objects | 80 | 82 | 79 | 5 / 3 | 0.73 | 3 / 4 | 1.00 |
| Language rewording | 89 | 85 | 88 | 1 / 5 | 0.22 | 2 / 3 | 1.00 |
| Lighting | 93 | 91 | 91 | 3 / 5 | 0.73 | 5 / 7 | 0.77 |
| Geometry (300) | 77.7 | 79.3 | 80.3 | 21 / 16 | 0.51 | 26 / 18 | 0.29 |
| Appearance (400) | 82.8 | 80.2 | 81.5 | 15 / 25 | 0.15 | 15 / 20 | 0.50 |
| All (700) | 80.6 | 79.9 | 81.0 | 36 / 41 | 0.65 | 41 / 38 | 0.82 |

B vs C, all 700: 34 / 42, p = 0.42; Object suite 83.4 vs 88.0, 2 / 10, p = 0.039 (the only
p < 0.05 among the 84 comparisons printed). A vs stock, all 700: 50 / 48, p = 0.92; start pose
68 vs 76, 4 / 12, p = 0.08. Full output: [test4/analysis/breakdown-2026-11-10.txt](test4/analysis/breakdown-2026-11-10.txt).

### Probe: held-out R² / cosine of a ridge regression from backbone features to the 1-s-ahead change in the V-JEPA 2 embedding

| Checkpoint | Layer the head read (input of the last decoder layer) | Layer 17 | Current embedding, head layer |
|---|---:|---:|---:|
| Stock | 0.210 / 0.435 | 0.270 / 0.498 | 0.845 |
| A | 0.208 / 0.433 | 0.268 / 0.497 | 0.844 |
| B | 0.334 / 0.502 | 0.291 / 0.507 | 0.857 |
| C | 0.272 / 0.461 | 0.269 / 0.499 | 0.887 |

Source: [test4/analysis/probe-2026-11-10.txt](test4/analysis/probe-2026-11-10.txt).
