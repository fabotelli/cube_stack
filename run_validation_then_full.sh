#!/usr/bin/env bash
# Two-stage driver:
#   STAGE A (validation): collect 800 successes-only eps -> train 12 epochs ->
#                         eval 100 eps. Cheap proof that the policy actually learns.
#   GATE: if the validation policy shows ANY real learning signal
#         (grasp_success >= 5%, or any place_on_top, or any overall_success),
#         it's "remotely ok" -> auto-continue. Otherwise STOP and flag.
#   STAGE B (full): hand off to run_pipeline.sh (collect 8000 -> train 25 ->
#                   finalize). Uses a SEPARATE dataset name (dataset_val.npz) so
#                   the full pipeline's skip-guard on dataset_cube_stack.npz is
#                   untouched and the real 8000-ep collection runs fresh.
set -uo pipefail

cd ~/mujoco-test/cube_stack
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MUJOCO_GL=osmesa

VAL_EPISODES=${VAL_EPISODES:-800}
VAL_EPOCHS=${VAL_EPOCHS:-12}
VAL_EVAL_EPS=${VAL_EVAL_EPS:-100}

echo "=== STAGE A: VALIDATION START $(date -u +%FT%TZ) eps=$VAL_EPISODES epochs=$VAL_EPOCHS ==="

# --- A1: collect 800 (separate name) ---
if [ -f dataset_val.npz ]; then
  echo "dataset_val.npz exists -> skipping validation collection"
else
  python3 collect_dataset_cube_stack.py \
      --episodes "$VAL_EPISODES" --workers 24 --resolution 256 --rate 50 \
      --max-frames-per-episode 160 --successes-only --seed 0 \
      --out dataset_val.npz > collect_val.log 2>&1
  echo "val collect rc=$? $(date -u +%FT%TZ)"; tail -8 collect_val.log
fi
[ -f dataset_val.npz ] || { echo "ABORT: no dataset_val.npz after collection"; exit 1; }

# --- A2: train 12 epochs ---
python3 train_bc_cube_stack.py \
    --data dataset_val.npz --out bc_cube_val.pt \
    --epochs "$VAL_EPOCHS" --lr 3e-4 --batch 256 --num-workers 8 --chunk 8 \
    --augment --ensemble-decay 0.01 > train_val.log 2>&1
echo "val train rc=$? $(date -u +%FT%TZ)"; tail -20 train_val.log
[ -f bc_cube_val.pt ] || { echo "ABORT: no bc_cube_val.pt after training"; exit 1; }

# --- A3: eval 100 eps ---
python3 eval_bc_cube_stack.py \
    --policy bc_cube_val.pt --episodes "$VAL_EVAL_EPS" \
    --start-seed 90000 --workers 24 --rate 50 --max-steps 170 \
    --ensemble-decay 0.01 --save-json eval_val.json > eval_val.log 2>&1
echo "val eval rc=$? $(date -u +%FT%TZ)"; cat eval_val.log

# --- GATE ---
echo "=== GATE $(date -u +%FT%TZ) ==="
GATE=$(python3 - <<'PY'
import json,sys
try:
    s=json.load(open("eval_val.json"))["policy_summary"]
except Exception as e:
    print("FAIL"); sys.stderr.write(f"gate: could not read eval_val.json: {e}\n"); sys.exit(0)
n=max(1,s.get("n",1))
grasp=s.get("grasp_success",0); place=s.get("place_on_top",0); overall=s.get("overall_success",0)
sys.stderr.write(f"gate: n={n} grasp={grasp} place_on_top={place} overall={overall} "
                 f"grasp_rate={grasp/n:.2%}\n")
# "remotely ok" = any real learning signal beyond the dead 0%-grasp smoke baseline
print("PASS" if (grasp/n>=0.05 or place>0 or overall>0) else "FAIL")
PY
)
echo "GATE RESULT: $GATE"

if [ "$GATE" != "PASS" ]; then
  {
    echo "VALIDATION GATE FAILED $(date -u +%FT%TZ)"
    echo "The 800-ep validation policy showed no real learning signal"
    echo "(grasp<5% and no place_on_top and no overall_success)."
    echo "Did NOT launch the full 8000-ep run. See eval_val.json / *_val.log."
  } > VALIDATION_FAILED.txt
  echo "STOP: validation did not pass the gate; see VALIDATION_FAILED.txt"
  exit 2
fi

# --- STAGE B: full pipeline (8000 -> train 25 -> finalize) ---
echo "=== GATE PASSED -> STAGE B: FULL PIPELINE $(date -u +%FT%TZ) ==="
bash run_pipeline.sh
echo "=== ALL DONE rc=$? $(date -u +%FT%TZ) ==="
