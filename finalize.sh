#!/usr/bin/env bash
# After training completes (bc_cube_stack.pt exists, no train process running),
# run the final 200-ep diagnostic eval, record a success demo, render the
# session notes from the eval JSON, and copy all deliverables into
# ~/Downloads/cube_stack_2026-06-10/.
set -uo pipefail

cd ~/mujoco-test/cube_stack

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MUJOCO_GL=osmesa

POLICY=bc_cube_stack.pt
EVAL_JSON=eval_cube_stack_final.json
DEMO=demo_cube_stack.mp4
NOTES=SESSION_NOTES_2026-06-10_cube_stack.md
DEST=~/Downloads/cube_stack_2026-06-10

if [ ! -f "$POLICY" ]; then
  echo "no $POLICY yet; bailing"
  exit 1
fi
if pgrep -f train_bc_cube_stack.py >/dev/null; then
  echo "training still running; bailing"
  exit 1
fi

echo "=== 200-ep diagnostic eval (24 workers) ==="
python3 eval_bc_cube_stack.py --policy "$POLICY" --episodes 200 \
    --start-seed 20000 --workers 24 --rate 50 --max-steps 170 \
    --ensemble-decay 0.01 --save-json "$EVAL_JSON" 2>&1 \
    | tee cube_stack_full_eval.log

echo "=== record success demo mp4 ==="
python3 record_bc_cube_stack.py --policy "$POLICY" --out "$DEMO" \
    --seed 20000 --max-search 30 --max-steps 170 --rate 50 \
    --ensemble-decay 0.01 --vid-res 512 --fps 30 2>&1 \
    | tee demo_record.log

echo "=== render session notes ==="
SOLVER_SELF_TEST=87 python3 render_notes.py "$NOTES"

echo "=== copy deliverables to $DEST ==="
mkdir -p "$DEST"
cp -f "$POLICY" "$EVAL_JSON" "$DEMO" "$NOTES" "$DEST/" 2>/dev/null || true
cp -f DECISIONS.md "$DEST/" 2>/dev/null || true
cp -f bc_cube_stack_epoch*.pt "$DEST/" 2>/dev/null || true
cp -f bc_cube_stack_eval100_epoch*.json "$DEST/" 2>/dev/null || true
cp -f bc_cube_stack_eval100_epoch*.log "$DEST/" 2>/dev/null || true
cp -f bc_cube_stack_inflight_eval.log "$DEST/" 2>/dev/null || true
cp -f train_cube_stack.log collect_cube_stack.log "$DEST/" 2>/dev/null || true
cp -f cube_stack_full_eval.log demo_record.log "$DEST/" 2>/dev/null || true
cp -f cube_stack_env.py cube_stack_solver.py collect_dataset_cube_stack.py \
   train_bc_cube_stack.py eval_bc_cube_stack.py record_bc_cube_stack.py \
   check_pixel_floor.py learm_scene_cube_stack.xml render_notes.py \
   finalize.sh run_pipeline.sh "$DEST/" 2>/dev/null || true
cp -rf pixel_floor_check "$DEST/" 2>/dev/null || true

echo "deliverables in $DEST:"
ls -la "$DEST"
echo "FINALIZE DONE"
