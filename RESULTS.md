# Does world-action-model training make a robot policy more robust?

Results as of 9 Nov 2026. Two open-weight LIBERO policies, same simulator, same tasks:

- **Alpha-S**: Griffin-Alpha-S-LIBERO (Qwen3-VL 4B backbone + flow-matching action expert). Plain VLA, no future prediction.
- **Fast-WAM**: lerobot/fastwam_libero_uncond_2cam224 (arXiv 2603.16666). Trained with a world-action objective.

Both score 94-100% on the standard LIBERO suites. The tests below change the conditions.

## Test 1: actions arrive late

Each action is applied d steps (0.1 s each) after the frame it was computed from. libero_object, 100 episodes per point.

| Delay (steps) | 0 | 2 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| Fast-WAM | 94 | 94 | 85 | 80 | 41 |
| Alpha-S, re-plan every step (K=1) | 99 | 93 | 95 | 81 | 50 |
| Alpha-S, re-plan every 10 steps (K=10) | 97 | 96 | 89 | 81 | 34 |

Both fall at the same rate. No delay advantage for the world-action model.

## Test 2: the target object moves

libero_object, 20 episodes per cell (2 start states x 10 tasks). The object is moved by 2, 5 or 10 cm
either before the first frame ("at the start") or after the arm has started reaching ("mid-reach").

| Move | 2 cm | 5 cm | 10 cm |
|---|---:|---:|---:|
| At the start: Alpha-S | 100 | 95 | 70 |
| At the start: Fast-WAM | 95 | 70 | 10 |
| Mid-reach: Alpha-S | 100 | 80 | 55 |
| Mid-reach: Fast-WAM | 100 | 60 | 0 |

No move: both 100%. Fast-WAM collapses at 10 cm (10% and 0%); Alpha-S keeps 55-70%.
A third mode ("blind": camera frozen after the move) scored 0% for both models at every size,
so it tests the camera freeze, not the move; it is dropped.

## Test 3: changed scenes (LIBERO-Plus)

700 LIBERO-Plus cases sampled across the four suites (perturbations: camera, lighting, layout,
distractors, language, robot init); 1 episode per case. 613 cases have both models so far.

| Suite | Cases | Alpha-S | Fast-WAM |
|---|---:|---:|---:|
| Spatial | 175 | 85.7% | 51.4% |
| Object | 175 | 88.0% | 71.4% |
| Goal | 175 | 70.3% | 40.0% |
| Long tasks, first half | 88 | 73.9% | 34.1% |
| **All 613** | 613 | **80.3%** | **51.4%** |

Fast-WAM's remaining 87 long-task cases: 55.2%. Alpha-S's are running.
Fast-WAM scores 100% on 10 unperturbed LIBERO cases with the same code path, so the drop is the perturbations.

## Test 4: add a future-prediction head to Alpha-S (running)

Fine-tune Alpha-S 6,000 steps on LIBERO three ways: A stock recipe; B + a head that predicts the
V-JEPA 2 embedding change 1 s ahead (loss weight 0.1); C + the same head predicting the current
embedding (control). Pilot (500 steps): 49 GB GPU memory, 1.65 s/step, future loss 1.0 -> 0.95.

## Caveats

- Tests 1-3 compare two different models (size, data, backbone), not one model with and without the
  world-action objective. Test 4 is the controlled version.
- Test 2 cells are 20 episodes: differences under ~25 points are within noise.
- Test 3's 613 cases are paired (same cases, both models): a 29-point gap is far outside chance.
