"""
record_bc_cube_stack.py — record a video of the chunked single-cam policy rolling
out the two-cube STACKING task.  Tries successive seeds from --seed until it finds
an episode the policy actually solves (farther cube stably stacked on nearer), then
writes that one.
"""

from __future__ import annotations

import os
import sys
if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import time
from collections import deque

import numpy as np
import torch
import mujoco
import imageio

from cube_stack_env import CubeStackEnv
from train_bc_cube_stack import BCPolicySideChunk


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


def rollout(env, model, mean, std, img_hw, k, seed, *, decay, rate, max_steps,
            vid_renderer, writer):
    import collect_dataset_cube_stack as _C
    env.reset(seed=seed)
    _C._dr_apply(env.model, np.random.default_rng(seed), _C._dr_setup(env.model))
    mujoco.mj_forward(env.model, env.data)

    pol_renderer_res = img_hw[0]
    env.model.vis.global_.offwidth = max(pol_renderer_res, vid_renderer.width)
    env.model.vis.global_.offheight = max(pol_renderer_res, vid_renderer.height)
    pol_renderer = mujoco.Renderer(env.model, height=pol_renderer_res,
                                   width=pol_renderer_res)

    qadr = _joint_qadr(env)
    arm_lo, arm_hi = env.arm_range[:, 0], env.arm_range[:, 1]
    grip_lo, grip_hi = env.model.actuator_ctrlrange[env.grip_act]

    chunks_buf: deque[np.ndarray] = deque(maxlen=k)
    for _ in range(max_steps):
        pol_renderer.update_scene(env.data, camera="policy_cam")
        s_full = pol_renderer.render().astype(np.float32) / 255.0
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
        env.step(rate)
        for cam in ("demo_cam", "policy_cam"):
            try:
                vid_renderer.update_scene(env.data, camera=cam)
                writer.append_data(vid_renderer.render())
                break
            except Exception:
                continue
    pol_renderer.close()
    return bool(env.is_stacked())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", default="demo_cube_stack.mp4")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--max-search", type=int, default=30,
                    help="how many sequential seeds to try until a success")
    ap.add_argument("--rate", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=170)
    ap.add_argument("--ensemble-decay", type=float, default=0.01)
    ap.add_argument("--vid-res", type=int, default=512)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    device = torch.device("cpu")
    model, mean, std, img_hw, k = _load_policy(args.policy, device)

    env = CubeStackEnv()
    env.model.vis.global_.offwidth = args.vid_res
    env.model.vis.global_.offheight = args.vid_res
    vid_renderer = mujoco.Renderer(env.model, height=args.vid_res, width=args.vid_res)

    found = False
    out_path = args.out
    for offset in range(args.max_search):
        seed = args.seed + offset
        candidate = out_path.replace(".mp4", f"_seed{seed}.mp4")
        writer = imageio.get_writer(candidate, fps=args.fps)
        t0 = time.perf_counter()
        ok = rollout(env, model, mean, std, img_hw, k, seed,
                     decay=args.ensemble_decay, rate=args.rate,
                     max_steps=args.max_steps, vid_renderer=vid_renderer,
                     writer=writer)
        writer.close()
        dt = time.perf_counter() - t0
        print(f"seed {seed}: {'SUCCESS' if ok else 'FAIL'}  ({dt:.1f}s) -> {candidate}",
              flush=True)
        if ok and not found:
            os.replace(candidate, out_path)
            print(f"  saved success rollout to {out_path}", flush=True)
            found = True
            break
        else:
            try:
                os.remove(candidate)
            except OSError:
                pass

    vid_renderer.close()
    if not found:
        env2 = CubeStackEnv()
        env2.model.vis.global_.offwidth = args.vid_res
        env2.model.vis.global_.offheight = args.vid_res
        vr2 = mujoco.Renderer(env2.model, height=args.vid_res, width=args.vid_res)
        writer = imageio.get_writer(out_path, fps=args.fps)
        rollout(env2, model, mean, std, img_hw, k, args.seed,
                decay=args.ensemble_decay, rate=args.rate,
                max_steps=args.max_steps, vid_renderer=vr2, writer=writer)
        writer.close()
        vr2.close()
        print(f"  no success in {args.max_search} seeds; recorded failure at "
              f"seed {args.seed} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
