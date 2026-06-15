#!/usr/bin/env bash
# One-shot driver for the two-cube stacking pipeline:
#   collect 8000 successes-only episodes -> train 25 epochs -> finalize
#   (200-ep diagnostic eval + demo + session notes + copy deliverables).
# Sequential so finalize's "is training still running?" guard sees a clean slate.
# Logs everything to pipeline.log.  Launch under nohup so it survives SSH drop.
set -uo pipefail

cd ~/mujoco-test/cube_stack

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MUJOCO_GL=osmesa

NPROC=$(nproc)
COLLECT_WORKERS=${COLLECT_WORKERS:-24}
EPISODES=${EPISODES:-8000}

echo "=== PIPELINE START $(date -u +%Y-%m-%dT%H:%M:%SZ)  nproc=$NPROC collect_workers=$COLLECT_WORKERS episodes=$EPISODES ==="

# --- 1) COLLECT ----------------------------------------------------------- #
if [ -f dataset_cube_stack.npz ]; then
  echo "dataset_cube_stack.npz already exists — skipping collection"
else
  echo "=== COLLECT ($EPISODES eps, $COLLECT_WORKERS workers, 256px, successes-only) ==="
  python3 collect_dataset_cube_stack.py \
      --episodes "$EPISODES" --workers "$COLLECT_WORKERS" \
      --resolution 256 --rate 50 --max-frames-per-episode 160 \
      --successes-only --out dataset_cube_stack.npz \
      > collect_cube_stack.log 2>&1
  COLLECT_RC=$?
  echo "collect exited rc=$COLLECT_RC at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  tail -8 collect_cube_stack.log
fi

if [ ! -f dataset_cube_stack.npz ]; then
  echo "no dataset_cube_stack.npz after collection — abort"; exit 1
fi

# --- 2) TRAIN ------------------------------------------------------------- #
echo "=== TRAIN (25 epochs, lr 3e-4, batch 256, workers 8, chunk 8, augment, decay 0.01) ==="
python3 train_bc_cube_stack.py \
    --data dataset_cube_stack.npz \
    --out bc_cube_stack.pt \
    --epochs 25 --lr 3e-4 --batch 256 \
    --num-workers 8 --chunk 8 --augment \
    --ensemble-decay 0.01 \
    > train_cube_stack.log 2>&1
TRAIN_RC=$?
echo "train exited rc=$TRAIN_RC at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
tail -30 train_cube_stack.log

if [ ! -f bc_cube_stack.pt ]; then
  echo "no bc_cube_stack.pt after training — abort before finalize"; exit 1
fi

# --- 3) FINALIZE ---------------------------------------------------------- #
echo "=== FINALIZE (200-ep diagnostic eval seed 20000 + demo mp4 + session notes) ==="
bash finalize.sh
echo "finalize exited rc=$? at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "=== PIPELINE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
