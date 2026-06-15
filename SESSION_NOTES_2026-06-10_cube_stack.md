# Session notes — 2026-06-10 (single-cam chunked BC for two-cube stacking)

Host: Lambda box — `~/mujoco-test/cube_stack/`

## TL;DR

- **`bc_cube_stack.pt` final 200-ep diagnostic eval (seed 20000, decay 0.01, max_steps 170, 24 workers):**
  - Overall success (farther cube stably stacked on nearer cube): **183/200 = 91.5%**
  - Correct-selection (picked the FARTHER cube, not the nearer): **199/200 = 99.5%**
  - Grasp success (farther cube lifted/moved): **199/200 = 99.5%**
  - Place-on-top (landed on the nearer cube): **183/200 = 91.5%**
  - Failure modes: wrong-cube-picked 1/200 = 0.5%, missed-grasp 0/200 = 0.0%, placement-miss 16/200 = 8.0%, knocked-over/unstable 0/200 = 0.0%
- **Verdict:** demo-ready.
- **Recipe is the proven single-cam chunked recipe verbatim** (k=8 action chunk,
  GELU encoder, AdamW 3e-4, batch 256, CosineAnnealingWarmRestarts T_0=25
  T_mult=1, GPU brightness/colour augment only, temporal-ensemble decay 0.01).
  Only the *task* + scene + solver are new.
- Solver baseline on the same 200 seeds: **182/200 = 91.0%** overall.

## Task

Two IDENTICAL red cubes, randomly placed each episode (non-overlapping, both in
reach).  Pick the cube FARTHER from the gripper HOME (x=0.145, y=0.0, workspace
plane) and stack it ON TOP of the NEARER cube.  Because the cubes are identical,
the policy gets NO appearance cue — it must localise both cubes from the single
256x256 side camera and infer farther-vs-nearer purely from position.  The
student sees only the side camera + joint state; the teacher uses ground-truth
positions.  GRIP and MOVE are separate steps.  Placement tolerance is tight
(~cube width = 0.018 m xy), so success needs precise placement, not a
forgiving bin drop.

## Pixel-floor preflight (256×256, two identical red cubes)

8 random scenes (seeds 9000–9007) + 6 forced worst-case pairs at the spawn
min-separation (0.065 m) across camera orientations.

| Stat | Value |
|---|---|
| cube bbox max side (px), min..max (mean) | 22..41 (mean 29.6) |
| Mean saturation of detected cube pixels | 0.83 |
| Circular-mean hue (deg) | 1.9 |
| Floor check (every cube ≥ 12 px) | PASS |
| Distinguishable (a background gap between the two red blobs, all scenes) | PASS |

The two cubes fuse into one red blob below ~0.055 m centre separation; spawn
min-sep was set to 0.065 m (worst-case background gap ~6 px) so they always
read as two objects.  Sample frames: `pixel_floor_check/pixel_floor_sample_*.png`,
`pixel_floor_check/min_sep_*.png`.

## Scene + solver (what's new vs the single-cube recipe)

- **Scene** `learm_scene_cube_stack.xml`: two identical red cubes (`cube_a`,
  `cube_b`, 1.9 cm side, free joints), no bins.  `CubeStackEnv.reset()`
  randomises both cube positions + yaws per episode with rejection-sampled
  non-overlap (min-sep 0.065 m).
- **Sequencing rule:** farther-from-home cube is picked up; nearer-from-home is
  the base.  Pure geometry (spawn positions), ties broken canonical (a, b).  The
  student must infer the same farther/nearer decision from pixels alone.
- **Solver** `cube_stack_solver.py`: single approach→descend→grasp→lift→
  over-base→lower→release→retreat FSM.  After the lift it reads the privileged
  grasp offset (held-cube centre relative to the pinch) and commands the pinch
  so the CUBE lands centred a 0.010 m gap above the base cube's top
  face — needed for the tight stack tolerance.  GRIP and MOVE are separate
  steps throughout.
- Solver 100-ep self-test (no policy, no DR): **87%** stacked.

## Data collection

```
collect_dataset_cube_stack.py --episodes 8000 --workers 24 \
    --resolution 256 --rate 50 --max-frames-per-episode 160 \
    --successes-only --out dataset_cube_stack.npz
```

- Single side camera; lighting + table DR only (cubes stay identical red).
- Result: kept **7270** episodes, **373051** frames, **73.36 GB**,
  in **2:49:50** wall time (solve rate 90.9%).

## Training

```
python3 train_bc_cube_stack.py --data dataset_cube_stack.npz \
    --out bc_cube_stack.pt \
    --epochs 25 --lr 3e-4 --batch 256 \
    --num-workers 8 --chunk 8 --augment --ensemble-decay 0.01
```

Architecture: `BCPolicySideChunk` — same ~13.2 M params as the proven recipe.
RAM plan: **full dataset fits in RAM**.  Train duration: **41m 39s**.

Per-epoch in-flight evals (100 ep, seed 20000, decay 0.01, max_steps 170):

| Epoch | Overall | Correct-sel | Grasp | Place-on-top |
|---|---|---|---|---|
| 5 | 76/100 = 76.0% | 95/100 = 95.0% | 95/100 = 95.0% | 76/100 = 76.0% |
| 10 | 84/100 = 84.0% | 96/100 = 96.0% | 96/100 = 96.0% | 84/100 = 84.0% |
| 15 | 88/100 = 88.0% | 96/100 = 96.0% | 96/100 = 96.0% | 88/100 = 88.0% |
| 20 | 86/100 = 86.0% | 99/100 = 99.0% | 99/100 = 99.0% | 86/100 = 86.0% |
| 25 | 91/100 = 91.0% | 99/100 = 99.0% | 99/100 = 99.0% | 91/100 = 91.0% |

## Final eval (200 ep, seed 20000, decay 0.01, max_steps 170, 24 workers)

| Metric | Value |
|---|---|
| Overall success | 183/200 = 91.5% |
| Correct-selection (picked FARTHER) | 199/200 = 99.5% |
| Grasp success | 199/200 = 99.5% |
| Place-on-top | 183/200 = 91.5% |
| Failure: wrong_cube_picked | 1/200 = 0.5% |
| Failure: missed_grasp | 0/200 = 0.0% |
| Failure: placement_miss | 16/200 = 8.0% |
| Failure: knocked_over_unstable | 0/200 = 0.0% |

JSON: `eval_cube_stack_final.json`.  Solver baseline on same 200 seeds:
**182/200 = 91.0%** overall.

## Deliverables

Remote `~/mujoco-test/cube_stack/`:
- `bc_cube_stack.pt` (best epoch checkpoint, **epoch 25 (val 0.00339)**)
- `bc_cube_stack_epoch{5,10,15,20,25}.pt` (periodic snapshots)
- `eval_cube_stack_final.json`
- `demo_cube_stack.mp4` (success rollout at seed 20000)
- Source files (`cube_stack_env.py`, `cube_stack_solver.py`,
  `collect_dataset_cube_stack.py`, `train_bc_cube_stack.py`,
  `eval_bc_cube_stack.py`, `record_bc_cube_stack.py`,
  `learm_scene_cube_stack.xml`, `check_pixel_floor.py`)
- Pixel-floor preflight: `pixel_floor_check/`
- Logs: `collect_cube_stack.log`, `train_cube_stack.log`,
  `cube_stack_full_eval.log`, `demo_record.log`

Copied to a dated downloads folder `~/Downloads/cube_stack_2026-06-10/`.

## Decisions logged (no-input run)

See `DECISIONS.md` for the assumptions made autonomously during this run
(min-separation choice, release-gap tuning, worker counts, eval step budget).
