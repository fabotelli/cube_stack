# Autonomous decisions — cube_stack run (2026-06-10)

Run executed under "no input" rules. Every decision point below was resolved
without asking; reasoning logged here.

1. **Cube identity / scene.** Two identical red cubes `cube_a`/`cube_b`, 1.9 cm,
   free joints, shared `cube_red` material. No bins (goal is stack, not drop).
   Mirrors the dice_sort scene minus the colour cue and mocap bins.

2. **Farther/nearer = SPAWN distance from home.** `dist_from_home` uses the
   cached spawn xy, not live xy. Reason: once a cube is lifted/stacked both
   cubes share xy, which would flip the order and break the success check.
   The task defines farther/nearer from the home position at episode start, so
   spawn-based is also the correct semantics. Home xy = (0.145, 0.0).

3. **Spawn min-separation = 0.065 m.** Pixel-floor preflight showed two
   identical red cubes FUSE into one blob below ~0.055 m centre separation when
   the separation vector aligns with the camera depth axis (worst-case bg gap
   0 px at 0.045). Swept min-sep: 0.045→0 px, 0.055→1 px, 0.065→6 px,
   0.075→10 px. Chose 0.065 (reliable ~6 px gap, all orientations) over 0.075
   to keep more of the reachable workspace usable. Did NOT drop resolution
   (128 failed in prior sessions). Cube zone x∈[0.130,0.185], y∈[-0.085,0.045].

4. **Release gap = 0.010 m (placement).** Initial 0.004 m gap shoved the base
   cube (gripper fingers reach base-top level and knock it; ~33% solve).
   Swept gap: 0.006→80%, 0.010→95%, 0.014→95%, 0.020→95%. Chose 0.010 (clears
   the fingertips, minimal drop height → least bounce). Placement uses the
   privileged grasp offset (held-cube centre vs pinch) so the CUBE, not the
   pinch, lands centred over the base — needed for the tight ~cube-width tol.

5. **Stack success tolerance.** STACK_XY_TOL = 0.018 m (~one cube width, tight
   as required), STACK_DZ_MIN = 0.012 m (top cube centre clearly above base;
   clean stack is ~0.019 m), base must still rest on the table.

6. **Collection workers = 24.** Task asked for "most of the cores" (~24 of 30).
   The dice_sort note found 24 over-subscribed, but that collector did NOT cap
   BLAS threads; this one exports OMP/MKL/OPENBLAS/NUMEXPR/VECLIB = 1 inside the
   collector module, so 24 osmesa-render workers should not thrash. Following
   the explicit task guidance.

7. **Eval workers.** In-flight per-epoch evals: 8 (concurrent with GPU
   training, plenty). Final standalone 200-ep eval: 24 (CPU-bound osmesa, no
   training competing) for speed.

8. **Eval step budget = 170.** Single pick-and-place is shorter than the
   dice_sort two-pick episode (which used 220). 170 steps × rate 50 × 0.002 s =
   17 s sim time, comfortably above the solver's per-episode length (verified in
   the smoke test). 

9. **Episodes = 8000, successes-only, 256px.** As specified. Teacher solve rate
   ~87% at min-sep 0.065 → expect ~6900 kept episodes. Trainer subsamples-to-fit
   RAM (box has ~216 GB free; full dataset expected ~90 GB → fits whole).

10. **Eval-JSON write hardened.** eval writes to `<out>.tmp`, fsync, then
    `os.replace` (atomic) — addresses the dice pipeline bug where the eval
    subprocess sometimes failed to leave a JSON. Verified in the smoke test
    before the full run.

11. **User message "wra for full" (mid-run).** Garbled/ambiguous; interpreted as
    "wrap up and go for the full run", consistent with the standing no-input
    directive. Proceeded with the full collect→train→finalize pipeline.

12. **Post-crash re-orientation (15:06Z).** Resumed with no memory under a briefing
    that claimed "collection COMPLETED, training crashed." Disk says otherwise and
    the briefing was followed-the-evidence-over-the-story:
    * The full pipeline is STILL RUNNING and healthy. `run_validation_then_full.sh`
      (PID 331022, parent=init/1 → detached, survived the disconnect) →
      `run_pipeline.sh` (341350) → `collect_dataset_cube_stack.py` (341353, 24
      workers). Validation already PASSED its gate (71% overall) at 14:52; STAGE B
      collection started 14:52:18 and is in progress (~600/8000 at 15:06, ETA ~2.6h).
      GPU idle is EXPECTED — we are in the CPU/osmesa collection phase; the GPU only
      engages once training starts.
    * The 24 `*.w{N}.images_side.npy` shards (~10.5 GB each, ~250 GB total) are NOT a
      finished dataset. Each is `open_memmap`-allocated at FULL CAPACITY
      (53,440 frames = 334 eps × 160 max-frames) at worker start; only the first
      `cursor` rows hold real (successes-only) data. No `.ckpt.npz` exist yet (first
      checkpoint is at ep 50/worker; workers were at ~25). No merged npz, and the
      `.run.json` manifest still present (merge deletes it on success). So "point the
      trainer at shards" is NOT viable — the trainer needs episode_starts/lengths/
      joint_names that only exist after merge_to_npz.
    * **The briefing's "250 GB > RAM → OOM" worry is a misread of the sparse shard
      allocation.** Real merged data ≈ 7,200 kept eps × ~51 frames × 196,608 B ≈
      ~80 GB (validation rate: 719 eps → 36,834 frames). Fits in 204 GB free; the
      trainer's auto-subsample (12 GB headroom, lines 220–241) is a safety net that
      likely won't trigger. DECISIONS #9 ("~90 GB → fits whole") was correct.
    * Decision: DO NOT kill/restart. The pipeline already runs the exact spec
      (train 25ep lr3e-4 batch256 workers8 chunk8 augment decay0.01; finalize =
      200-ep eval seed20000 24w + render_notes.py → SESSION_NOTES_2026-06-10). Let
      it run; verify GPU at train time via a background watcher; report at the end.
