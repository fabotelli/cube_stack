"""
eval_bc_cube_stack.py
=====================

Parallel held-out eval for chunked single-cam BC on the two-cube STACKING task.
Same ACT temporal-ensemble inference as eval_bc_dice_sort.py, but reports the
stacking diagnostic breakdown requested for this task:

  * overall success            : farther cube stably stacked on the nearer cube
                                  (env.is_stacked, privileged, spawn-order).
  * correct-selection rate     : did it manipulate the FARTHER cube (not the
                                  nearer one)?  Isolates the spatial-reasoning
                                  capability from the manipulation.
  * grasp success              : was the farther cube actually picked up / moved?
  * place-on-top success       : did the carried cube land on the nearer cube
                                  (xy within tol AND clearly above it)?
  * failure-mode counts (one primary label per episode):
        wrong_cube_picked      : lifted / stacked the NEARER cube instead.
        missed_grasp           : neither cube moved (never grasped).
        placement_miss         : picked the farther cube but it ended off the
                                  base (beside it, not stacked).
        knocked_over_unstable  : landed near the base but not stably stacked
                                  (toppled, or the base got knocked).
"""

from __future__ import annotations

import os
import sys
if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time
from collections import deque

import numpy as np
import torch
import mujoco

from cube_stack_env import (CubeStackEnv, STACK_XY_TOL, STACK_DZ_MIN, CUBES)
from cube_stack_solver import CubeStackSolver
from train_bc_cube_stack import BCPolicySideChunk


MOVE_TOL = 0.020   # xy displacement from spawn that counts as "cube manipulated"
ELEV_Z = 0.020     # cube centre this far above table top counts as "lifted"


def _load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["arch"]
    model = BCPolicySideChunk(n_joints=a["n_joints"], in_ch=a["in_ch"],
                              chunk=a["chunk"], img_hw=tuple(a["img_hw"]),
                              bottleneck=a["bottleneck"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    mean = np.asarray(ck["joint_mean"], dtype=np.float32)
    std = np.asarray(ck["joint_std"], dtype=np.float32)
    return model, mean, std, tuple(a["img_hw"]), int(a["chunk"])


def _joint_qadr(env):
    grip_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "grip_left")
    return np.append(env.arm_qpos, env.model.jnt_qposadr[grip_jid]).astype(np.intp)


def _make_renderer(env, res):
    env.model.vis.global_.offwidth = res
    env.model.vis.global_.offheight = res
    return mujoco.Renderer(env.model, height=res, width=res)


def _classify_episode(env):
    """Classify the final scene state into the stacking diagnostic booleans plus
    a single primary outcome label.  Uses privileged spawn-order farther/nearer."""
    farther, nearer = env.pick_order()
    table = env.table_top_z
    fp = env.cube_pos(farther)
    pn = env.cube_pos(nearer)
    fs = env.cube_xy0[farther]
    ns = env.cube_xy0[nearer]

    far_dxy = float(np.linalg.norm(fp[:2] - fs))
    near_dxy = float(np.linalg.norm(pn[:2] - ns))
    far_dz = float(fp[2] - table)
    near_dz = float(pn[2] - table)

    far_moved = far_dxy > MOVE_TOL or far_dz > ELEV_Z
    near_moved = near_dxy > MOVE_TOL or near_dz > ELEV_Z
    far_elev = far_dz > ELEV_Z
    near_elev = near_dz > ELEV_Z

    dxy_fn = float(np.linalg.norm(fp[:2] - pn[:2]))
    place_on_top = dxy_fn < STACK_XY_TOL and (fp[2] - pn[2]) > STACK_DZ_MIN
    stacked_inverted = (dxy_fn < STACK_XY_TOL and (pn[2] - fp[2]) > STACK_DZ_MIN
                        and not far_elev)

    success = bool(env.is_stacked())
    grasp_success = bool(far_moved)
    correct_selection = bool(far_moved and not near_elev)

    if success:
        label = "success"
    elif stacked_inverted or (near_elev and not far_elev):
        label = "wrong_cube_picked"
    elif not far_moved and not near_moved:
        label = "missed_grasp"
    elif far_moved and dxy_fn >= STACK_XY_TOL:
        label = "placement_miss"
    else:
        label = "knocked_over_unstable"

    return dict(farther=farther, nearer=nearer,
                success=success, grasp_success=grasp_success,
                correct_selection=correct_selection, place_on_top=bool(place_on_top),
                label=label,
                far_final=fp.tolist(), near_final=pn.tolist(),
                dxy_fn=dxy_fn)


def eval_chunk(task):
    wid, seeds, cfg, gl_lock = task
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model, mean, std, img_hw, k = _load_policy(cfg["policy_path"], device)
    env = CubeStackEnv()
    with gl_lock:
        renderer = _make_renderer(env, img_hw[0])
    qadr = _joint_qadr(env)
    import collect_dataset_cube_stack as _C
    dr_state = _C._dr_setup(env.model)
    arm_lo, arm_hi = env.arm_range[:, 0], env.arm_range[:, 1]
    grip_lo, grip_hi = env.model.actuator_ctrlrange[env.grip_act]
    decay = float(cfg["ensemble_decay"])

    results = []
    for seed in seeds:
        env.reset(seed=seed)
        _C._dr_apply(env.model, np.random.default_rng(seed), dr_state)
        mujoco.mj_forward(env.model, env.data)
        chunks_buf: deque[np.ndarray] = deque(maxlen=k)

        for _ in range(cfg["eval_max_steps"]):
            renderer.update_scene(env.data, camera="policy_cam")
            s_full = renderer.render().astype(np.float32) / 255.0
            s_t = torch.from_numpy(s_full).permute(2, 0, 1).contiguous().unsqueeze(0)
            joints = env.data.qpos[qadr].astype(np.float32)
            jin = torch.from_numpy((joints - mean) / std).unsqueeze(0)
            with torch.no_grad():
                pred_chunk = model(s_t, jin).squeeze(0).numpy()
            pred_chunk = pred_chunk * std + mean

            chunks_buf.append(pred_chunk)
            n = len(chunks_buf)
            weights = np.exp(-decay * np.arange(n)[::-1])
            weights /= weights.sum()
            stacked = np.stack([c[n - 1 - i] for i, c in enumerate(chunks_buf)])
            action = (weights[:, None] * stacked).sum(axis=0)

            env.set_arm_target(np.clip(action[:5], arm_lo, arm_hi))
            env.set_gripper(float(np.clip(action[5], grip_lo, grip_hi)))
            env.step(cfg["rate"])

        cls = _classify_episode(env)
        cls["seed"] = int(seed)
        results.append(cls)
    renderer.close()
    return results


def solver_chunk(task):
    wid, seeds, cfg, gl_lock = task
    env = CubeStackEnv()
    solver = CubeStackSolver(env)
    results = []
    for seed in seeds:
        env.reset(seed=seed)
        solver.run_episode()
        cls = _classify_episode(env)
        cls["seed"] = int(seed)
        results.append(cls)
    return results


def _split(seeds, w):
    return [seeds[i::w] for i in range(w)]


def _parallel(fn, seeds, cfg, workers, gl_lock, pool):
    tasks = [(i, c, cfg, gl_lock) for i, c in enumerate(_split(seeds, workers))]
    return [r.get() for r in [pool.apply_async(fn, (t,)) for t in tasks]]


def _summarise(rows):
    n = len(rows)
    labels = ["success", "wrong_cube_picked", "missed_grasp",
              "placement_miss", "knocked_over_unstable"]
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in labels}
    return dict(
        n=n,
        overall_success=sum(r["success"] for r in rows),
        correct_selection=sum(r["correct_selection"] for r in rows),
        grasp_success=sum(r["grasp_success"] for r in rows),
        place_on_top=sum(r["place_on_top"] for r in rows),
        wrong_cube_picked=counts["wrong_cube_picked"],
        missed_grasp=counts["missed_grasp"],
        placement_miss=counts["placement_miss"],
        knocked_over_unstable=counts["knocked_over_unstable"],
    )


def _percent(s, n):
    return f"{s}/{n} = {100*s/n:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--start-seed", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=170)
    ap.add_argument("--rate", type=int, default=50)
    ap.add_argument("--ensemble-decay", type=float, default=0.01)
    ap.add_argument("--save-json", default=None)
    ap.add_argument("--skip-solver", action="store_true")
    args = ap.parse_args()

    cfg = dict(rate=args.rate, eval_max_steps=args.max_steps,
               policy_path=args.policy, ensemble_decay=args.ensemble_decay)
    seeds = [args.start_seed + i for i in range(args.episodes)]

    t0 = time.perf_counter()
    with mp.Manager() as mgr:
        gl = mgr.Lock()
        with mp.Pool(args.workers) as pool:
            print(f"running {args.episodes} eval episodes on {args.workers} "
                  f"workers (decay={args.ensemble_decay}, max_steps={args.max_steps})...",
                  flush=True)
            pol_chunks = _parallel(eval_chunk, seeds, cfg, args.workers, gl, pool)
            pol_rows = [r for chunk in pol_chunks for r in chunk]
            print(f"  policy eval done in {time.perf_counter()-t0:.1f}s", flush=True)
            if not args.skip_solver:
                sol_chunks = _parallel(solver_chunk, seeds, cfg, args.workers, gl, pool)
                sol_rows = [r for chunk in sol_chunks for r in chunk]
            else:
                sol_rows = []
    el = time.perf_counter() - t0
    pol_sum = _summarise(pol_rows)
    sol_sum = _summarise(sol_rows) if sol_rows else None
    n = args.episodes

    print("\n" + "=" * 70)
    print(f"  {args.policy:35s}  ({n} eps, decay {args.ensemble_decay})")
    print(f"  overall success (farther stacked on nearer) : {_percent(pol_sum['overall_success'], n)}")
    print(f"  correct-selection (picked FARTHER cube)      : {_percent(pol_sum['correct_selection'], n)}")
    print(f"  grasp success (farther cube lifted/moved)    : {_percent(pol_sum['grasp_success'], n)}")
    print(f"  place-on-top (landed on the nearer cube)     : {_percent(pol_sum['place_on_top'], n)}")
    print(f"  -- failure modes --")
    print(f"  wrong_cube_picked     : {_percent(pol_sum['wrong_cube_picked'], n)}")
    print(f"  missed_grasp          : {_percent(pol_sum['missed_grasp'], n)}")
    print(f"  placement_miss        : {_percent(pol_sum['placement_miss'], n)}")
    print(f"  knocked_over_unstable : {_percent(pol_sum['knocked_over_unstable'], n)}")
    if sol_sum is not None:
        print(f"  -- solver baseline overall : {_percent(sol_sum['overall_success'], n)}")
    print(f"  elapsed {el:.1f}s")
    print("=" * 70, flush=True)

    out = args.save_json
    if out:
        payload = dict(policy=args.policy, episodes=args.episodes,
                       start_seed=args.start_seed,
                       ensemble_decay=args.ensemble_decay,
                       max_steps=args.max_steps, rate=args.rate,
                       elapsed_s=el, policy_summary=pol_sum,
                       solver_summary=sol_sum,
                       policy_rows=pol_rows, solver_rows=sol_rows)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=lambda o: o.tolist()
                      if hasattr(o, "tolist") else float(o))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
