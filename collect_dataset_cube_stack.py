"""
collect_dataset_cube_stack.py
=============================

Robot-legal imitation dataset for the two-cube STACKING task.  Same parallel /
checkpointed / merge-into-npz pipeline as ``collect_dataset_dice_sort.py`` but
driving ``CubeStackEnv`` + ``CubeStackSolver`` (a single pick-and-place per
episode: pick the farther cube, stack it on the nearer cube).

Differences vs the dice-sort collector:
  * Imports CubeStackEnv / CubeStackSolver.
  * Domain randomisation: lighting + table only.  The cubes are identical red on
    purpose (no colour cue), so cube material is never jittered.
  * Single side camera (no wrist channel).
  * Per-episode privileged metadata is (farther, nearer, success) instead of the
    dice-sort colour/bin labels.
"""

from __future__ import annotations

import os
import sys
if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
# Single-thread BLAS so N osmesa CPU workers don't thrash each other.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import multiprocessing as mp
import time

import numpy as np
import mujoco

from cube_stack_env import CubeStackEnv, ARM_JOINTS
from cube_stack_solver import CubeStackSolver


CAM_LOOKAT = np.array([0.12, 0.04, 0.13])
CAM_DISTANCE = 0.62
CAM_AZIMUTH = 128.0
CAM_ELEVATION = -20.0

JOINT_NAMES = ARM_JOINTS + ["grip_left"]
N_JOINTS = len(JOINT_NAMES)


class DatasetLogger:
    """Side-camera + joint-angle capture (no wrist branch)."""

    def __init__(self, env, *, capacity, rate, height, width,
                 side_img_path, joints_path,
                 resume=False, start_cursor=0, action_noise=0.0):
        self.env = env
        self.rate = rate
        self.capacity = capacity
        self.action_noise = float(action_noise)

        env.model.vis.global_.offwidth = width
        env.model.vis.global_.offheight = height

        self.renderer = mujoco.Renderer(env.model, height=height, width=width)

        m = env.model
        grip_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "grip_left")
        self.joint_qadr = np.append(
            env.arm_qpos, m.jnt_qposadr[grip_jid]).astype(np.intp)

        if resume:
            self.images_side = np.lib.format.open_memmap(side_img_path, mode="r+")
            self.joints = np.lib.format.open_memmap(joints_path, mode="r+")
        else:
            self.images_side = np.lib.format.open_memmap(
                side_img_path, mode="w+", dtype=np.uint8,
                shape=(capacity, height, width, 3))
            self.joints = np.lib.format.open_memmap(
                joints_path, mode="w+", dtype=np.float32,
                shape=(capacity, len(self.joint_qadr)))

        self.cursor = start_cursor
        self._substep = 0
        self.overflowed = False

    def start_episode(self) -> int:
        self._substep = 0
        return self.cursor

    def rewind(self, start: int) -> None:
        self.cursor = start

    def capture(self, data) -> None:
        if self._substep % self.rate == 0:
            if self.cursor >= self.capacity:
                self.overflowed = True
            else:
                self.renderer.update_scene(data, camera="policy_cam")
                self.images_side[self.cursor] = self.renderer.render()
                j = data.qpos[self.joint_qadr].astype(np.float32)
                if self.action_noise > 0.0:
                    j = j + np.random.randn(len(j)).astype(np.float32) * self.action_noise
                self.joints[self.cursor] = j
                self.cursor += 1
        self._substep += 1

    def flush(self) -> None:
        self.images_side.flush()
        self.joints.flush()

    def close(self) -> None:
        self.renderer.close()


# --------------------------------------------------------------------------- #
#  Domain randomisation: lighting + table brightness.  Cubes stay identical red.
# --------------------------------------------------------------------------- #
def _dr_setup(model):
    mat = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, n)
    return dict(tab_mat=mat("table"),
                base_light=model.light_diffuse.copy(),
                base_tab=model.mat_rgba[mat("table")].copy())


def _dr_apply(model, rng, s):
    model.light_diffuse[:] = np.clip(
        s["base_light"] * rng.uniform(0.6, 1.4, size=s["base_light"].shape),
        0.0, 1.0)
    model.mat_rgba[s["tab_mat"], :3] = np.clip(
        s["base_tab"][:3] * rng.uniform(0.8, 1.2), 0.0, 1.0)


def fmt_hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _close_mmap(arr) -> None:
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        try:
            mm.close()
        except Exception:
            pass


def _rm(path) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _write_checkpoint(path, i_done, cursor, lengths, seeds, success,
                      n_success, farther, nearer):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, i_done=np.int64(i_done), cursor=np.int64(cursor),
                 lengths=np.asarray(lengths, np.int64),
                 seeds=np.asarray(seeds, np.int64),
                 success=np.asarray(success, bool),
                 n_success=np.int64(n_success),
                 farther=np.asarray(farther),
                 nearer=np.asarray(nearer))
    os.replace(tmp, path)


def _load_checkpoint(path):
    with np.load(path) as ck:
        return dict(i_done=int(ck["i_done"]), cursor=int(ck["cursor"]),
                    lengths=ck["lengths"].tolist(),
                    seeds=ck["seeds"].tolist(),
                    success=ck["success"].tolist(),
                    n_success=int(ck["n_success"]),
                    farther=[str(s) for s in ck["farther"].tolist()],
                    nearer=[str(s) for s in ck["nearer"].tolist()])


def collect_chunk(task):
    worker_id, ep_start, ep_count, base_seed, cfg, progress, gl_lock = task

    side_path = cfg["side_img_tmpl"].format(w=worker_id)
    jnt_path = cfg["jnt_tmpl"].format(w=worker_id)
    ckpt_path = cfg["ckpt_tmpl"].format(w=worker_id)

    i_start, cursor0, n_success = 0, 0, 0
    lengths, seeds, success = [], [], []
    farther, nearer = [], []
    resume = False
    if (cfg["resume"] and os.path.exists(ckpt_path)
            and os.path.exists(side_path) and os.path.exists(jnt_path)):
        try:
            ck = _load_checkpoint(ckpt_path)
            i_start, cursor0, n_success = ck["i_done"], ck["cursor"], ck["n_success"]
            lengths, seeds, success = ck["lengths"], ck["seeds"], ck["success"]
            farther, nearer = ck["farther"], ck["nearer"]
            resume = True
        except Exception:
            i_start, cursor0, n_success = 0, 0, 0
            lengths, seeds, success = [], [], []
            farther, nearer = [], []
            resume = False

    env = CubeStackEnv()
    with gl_lock:
        logger = DatasetLogger(
            env, capacity=ep_count * cfg["max_frames"], rate=cfg["rate"],
            height=cfg["res"], width=cfg["res"],
            side_img_path=side_path, joints_path=jnt_path,
            resume=resume, start_cursor=cursor0,
            action_noise=cfg.get("action_noise", 0.0))
    solver = CubeStackSolver(env, recorder=logger)
    dr = _dr_setup(env.model)

    progress[worker_id] = i_start
    last_ckpt = i_start
    for i in range(i_start, ep_count):
        seed = base_seed + ep_start + i
        env.reset(seed=seed)
        _dr_apply(env.model, np.random.default_rng(seed), dr)
        start = logger.start_episode()
        log = solver.run_episode()
        s = bool(log["success"])
        n_success += s
        if cfg["successes_only"] and not s:
            logger.rewind(start)
        else:
            lengths.append(logger.cursor - start)
            seeds.append(seed)
            success.append(s)
            farther.append(log["farther"])
            nearer.append(log["nearer"])
        progress[worker_id] = i + 1
        if (i + 1) - last_ckpt >= cfg["checkpoint_every"]:
            logger.flush()
            _write_checkpoint(ckpt_path, i + 1, logger.cursor,
                              lengths, seeds, success, n_success, farther, nearer)
            last_ckpt = i + 1

    logger.flush()
    _write_checkpoint(ckpt_path, ep_count, logger.cursor,
                      lengths, seeds, success, n_success, farther, nearer)
    logger.close()
    return dict(worker_id=worker_id, n_frames=logger.cursor,
                side_path=side_path, jnt_path=jnt_path, ckpt_path=ckpt_path,
                lengths=lengths, seeds=seeds, success=success,
                farther=farther, nearer=nearer,
                n_run=ep_count, n_success=n_success, overflowed=logger.overflowed)


def merge_to_npz(results, out_path, cfg, timestep, compress):
    results = sorted(results, key=lambda r: r["worker_id"])
    total = sum(r["n_frames"] for r in results)
    res = cfg["res"]

    merged_side = np.lib.format.open_memmap(
        cfg["merge_side"], mode="w+", dtype=np.uint8, shape=(total, res, res, 3))
    merged_jnt = np.lib.format.open_memmap(
        cfg["merge_jnt"], mode="w+", dtype=np.float32, shape=(total, N_JOINTS))

    lengths, seeds, success, farther, nearer = [], [], [], [], []
    off, CH = 0, 4096
    for r in results:
        n = r["n_frames"]
        ws = np.load(r["side_path"], mmap_mode="r")
        wj = np.load(r["jnt_path"], mmap_mode="r")
        for i in range(0, n, CH):
            j = min(i + CH, n)
            merged_side[off + i:off + j] = ws[i:j]
            merged_jnt[off + i:off + j] = wj[i:j]
        off += n
        lengths += r["lengths"]
        seeds += r["seeds"]
        success += r["success"]
        farther += r["farther"]
        nearer += r["nearer"]
        _close_mmap(ws); _close_mmap(wj)
        del ws, wj
    merged_side.flush()
    merged_jnt.flush()
    for r in results:
        _rm(r["side_path"])
        _rm(r["jnt_path"])
        _rm(r["ckpt_path"])

    lengths = np.array(lengths, dtype=np.int64)
    starts = (np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
              if len(lengths) else np.empty(0, np.int64))
    episode_id = (np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
                  if len(lengths) else np.empty(0, np.int32))

    payload = dict(
        images_side=merged_side,
        joint_angles=merged_jnt,
        joint_names=np.array(JOINT_NAMES),
        episode_id=episode_id, episode_starts=starts, episode_lengths=lengths,
        episode_seeds=np.array(seeds, dtype=np.int64),
        episode_success=np.array(success, dtype=bool),
        episode_farther=np.array(farther),
        episode_nearer=np.array(nearer),
        subsample_rate=np.int64(cfg["rate"]),
        physics_timestep=np.float64(timestep),
        image_hw=np.array([res, res], dtype=np.int64),
        cam_lookat=CAM_LOOKAT, cam_distance=np.float64(CAM_DISTANCE),
        cam_azimuth=np.float64(CAM_AZIMUTH), cam_elevation=np.float64(CAM_ELEVATION),
    )
    (np.savez_compressed if compress else np.savez)(out_path, **payload)

    del payload
    _close_mmap(merged_side); _close_mmap(merged_jnt)
    del merged_side, merged_jnt
    _rm(cfg["merge_side"]); _rm(cfg["merge_jnt"])
    return total


def _run_signature(args):
    return dict(episodes=args.episodes, workers=args.workers, seed=args.seed,
                rate=args.rate, resolution=args.resolution,
                max_frames=args.max_frames_per_episode,
                successes_only=bool(args.successes_only),
                out=os.path.basename(args.out))


def _purge_temps(cfg, manifest):
    for pat in (cfg["side_img_tmpl"], cfg["jnt_tmpl"], cfg["ckpt_tmpl"]):
        for p in glob.glob(pat.replace("{w}", "*")):
            _rm(p)
        _rm(pat.replace("{w}", "*") + ".tmp")
    _rm(cfg["merge_side"])
    _rm(cfg["merge_jnt"])
    _rm(manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=int, default=50)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--successes-only", action="store_true", default=True)
    ap.add_argument("--no-successes-only", action="store_false",
                    dest="successes_only")
    ap.add_argument("--max-frames-per-episode", type=int, default=160)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--out", default="dataset_cube_stack.npz")
    ap.add_argument("--tmpdir", default=None)
    ap.add_argument("--merge-tmpdir", default=None)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--action-noise", type=float, default=0.0)
    args = ap.parse_args()

    if args.episodes <= 0:
        ap.error("--episodes must be positive")
    if args.workers <= 0 or args.workers > args.episodes:
        ap.error("--workers must be in 1..episodes")
    if args.rate <= 0:
        ap.error("--rate must be positive")
    if args.resolution <= 0 or args.resolution > 720:
        ap.error("--resolution must be in 1..720")
    if args.max_frames_per_episode <= 0 or args.checkpoint_every <= 0:
        ap.error("--max-frames-per-episode and --checkpoint-every must be positive")

    res, W, E = args.resolution, args.workers, args.episodes

    chunks, start = [], 0
    for w in range(W):
        cnt = E // W + (1 if w < E % W else 0)
        chunks.append((w, start, cnt))
        start += cnt

    out_dir = os.path.dirname(os.path.abspath(args.out))
    tmpdir = args.tmpdir or out_dir
    os.makedirs(tmpdir, exist_ok=True)
    merge_dir = args.merge_tmpdir or tmpdir
    os.makedirs(merge_dir, exist_ok=True)
    base = os.path.basename(args.out)
    manifest = os.path.join(out_dir, base + ".run.json")
    cfg = dict(
        rate=args.rate, res=res, max_frames=args.max_frames_per_episode,
        successes_only=args.successes_only, checkpoint_every=args.checkpoint_every,
        action_noise=args.action_noise,
        side_img_tmpl=os.path.join(tmpdir, base + ".w{w}.images_side.npy"),
        jnt_tmpl=os.path.join(tmpdir, base + ".w{w}.joints.npy"),
        ckpt_tmpl=os.path.join(tmpdir, base + ".w{w}.ckpt.npz"),
        merge_side=os.path.join(merge_dir, base + ".merge.images_side.npy"),
        merge_jnt=os.path.join(merge_dir, base + ".merge.joints.npy"),
    )

    sig = _run_signature(args)
    if args.fresh:
        _purge_temps(cfg, manifest)
    resuming = False
    if os.path.exists(manifest):
        try:
            with open(manifest) as f:
                saved = json.load(f)
        except Exception:
            saved = None
        if saved == sig:
            resuming = True
        else:
            ap.error(f"{manifest} is from a different config; pass --fresh to "
                     f"restart or use matching arguments.")
    if not resuming:
        _purge_temps(cfg, manifest)
        with open(manifest, "w") as f:
            json.dump(sig, f, indent=2)
    cfg["resume"] = resuming

    timestep = float(CubeStackEnv().model.opt.timestep)

    per_worker_gb = (max(c for _, _, c in chunks) * args.max_frames_per_episode
                     * res * res * 3 / 1e9)
    print(f"{'RESUMING' if resuming else 'Collecting'} {E} episodes across {W} "
          f"workers @ {res}x{res} px, every {args.rate} substeps "
          f"(~{500.0 / args.rate:.0f} Hz)."
          f"{'  [successful only]' if args.successes_only else ''}", flush=True)
    print(f"Per-worker episodes: {[c for _, _, c in chunks]}; checkpoint every "
          f"{args.checkpoint_every}; temps in {tmpdir} (~{per_worker_gb:.1f} GB/worker).",
          flush=True)

    t0 = time.perf_counter()
    with mp.Manager() as manager:
        progress = manager.list([0] * W)
        gl_lock = manager.Lock()
        with mp.Pool(W) as pool:
            tasks = [(w, s, c, args.seed, cfg, progress, gl_lock)
                     for (w, s, c) in chunks]
            async_results = [pool.apply_async(collect_chunk, (t,)) for t in tasks]
            pool.close()

            done0, last_bucket = None, -1
            while True:
                ready = all(r.ready() for r in async_results)
                done = sum(progress)
                if done0 is None:
                    done0 = done
                bucket = done // args.progress_every
                if bucket != last_bucket or ready:
                    elapsed = time.perf_counter() - t0
                    eps = (done - done0) / elapsed if elapsed > 0 else 0.0
                    eta = (E - done) / eps if eps > 0 else 0.0
                    print(f"  [{done:>6d}/{E}] workers={list(progress)} "
                          f"{eps:4.2f} ep/s  elapsed {fmt_hms(elapsed)}  "
                          f"ETA {fmt_hms(eta)}", flush=True)
                    last_bucket = bucket
                if ready:
                    break
                time.sleep(5)

            results = [r.get() for r in async_results]
            pool.join()

    n_run = sum(r["n_run"] for r in results)
    n_success = sum(r["n_success"] for r in results)
    n_kept = sum(len(r["lengths"]) for r in results)
    if any(r["overflowed"] for r in results):
        print("WARNING: a worker hit its frame capacity; some tail frames were "
              "dropped. Re-run --fresh with a larger --max-frames-per-episode.",
              flush=True)

    print(f"\nAll workers done in {fmt_hms(time.perf_counter() - t0)}. "
          f"Kept {n_kept} episodes. Merging into {args.out} ...", flush=True)
    total = merge_to_npz(results, args.out, cfg, timestep, args.compress)
    _rm(manifest)

    size_gb = os.path.getsize(args.out) / 1e9
    print(f"Saved {args.out} ({size_gb:.2f} GB) in "
          f"{fmt_hms(time.perf_counter() - t0)}: {total} frames from {n_kept} "
          f"episodes ({100.0 * n_success / max(n_run, 1):.1f}% solve rate).",
          flush=True)


if __name__ == "__main__":
    main()
