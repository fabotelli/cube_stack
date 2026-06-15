"""
train_bc_cube_stack.py
======================

Identical to train_bc_dice_sort.py / train_bc_sidecam_chunk.py (same
architecture, optimizer, schedule and augmentation) -- the proven single-cam
chunked recipe that hit 100% on pick-and-place and ~76% on dice-sort.  Only the
dataset and the periodic in-flight eval target (eval_bc_cube_stack.py) differ.

Recipe (do NOT change): single side cam 256x256, GELU encoder, chunk=8, temporal
ensemble decay 0.01, AdamW 3e-4, batch 256, CosineAnnealingWarmRestarts
(T_0=epochs, T_mult=1), GPU brightness/colour augment only (no Gaussian noise).
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import subprocess
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def npz_member_memmap(npz_path: str, member: str) -> np.memmap:
    with zipfile.ZipFile(npz_path) as zf:
        zi = zf.getinfo(member)
        if zi.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{member} is compressed; use savez (not savez_compressed).")
        hdr_off = zi.header_offset
    with open(npz_path, "rb") as f:
        f.seek(hdr_off)
        local = f.read(30)
        name_len = int.from_bytes(local[26:28], "little")
        extra_len = int.from_bytes(local[28:30], "little")
        f.seek(hdr_off + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        np.lib.format._check_version(version)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        data_off = f.tell()
    return np.memmap(npz_path, mode="r", dtype=dtype, shape=shape,
                     offset=data_off, order="F" if fortran else "C")


class RAMSideChunkDataset(Dataset):
    """Returns (img_side, joints_now, action_chunk[k, n_joints])."""

    def __init__(self, side, frame_idx, joints_in_norm, actions_norm):
        self.side = side
        self.frame_idx = frame_idx
        self.joints_in = joints_in_norm
        self.actions = actions_norm

    def __len__(self):
        return len(self.frame_idx)

    def __getitem__(self, idx):
        i = int(self.frame_idx[idx])
        s = self.side[i].astype(np.float32) / 255.0
        s_t = torch.from_numpy(s).permute(2, 0, 1).contiguous()
        jin = torch.from_numpy(self.joints_in[idx])
        act = torch.from_numpy(self.actions[idx])
        return s_t, jin, act


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    B = x.size(0)
    bright = (1.0 + torch.randn(B, 1, 1, 1, device=x.device) * 0.15).clamp(0.7, 1.3)
    color = torch.empty(B, 3, 1, 1, device=x.device).uniform_(-0.05, 0.05)
    return (x * bright + color).clamp_(0.0, 1.0)


class CamEncoder(nn.Module):
    def __init__(self, in_ch=3, img_hw=(256, 256), bottleneck=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.GELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, *img_hw)
            flat_dim = self.cnn(dummy).shape[1]
        self.bottleneck = nn.Sequential(
            nn.Linear(flat_dim, bottleneck), nn.GELU())
        self.feat_dim = bottleneck

    def forward(self, x):
        return self.bottleneck(self.cnn(x))


class BCPolicySideChunk(nn.Module):
    """Outputs (B, chunk, n_joints)."""

    def __init__(self, n_joints=6, in_ch=3, chunk=8,
                 img_hw=(256, 256), bottleneck=256, hidden=(512, 256)):
        super().__init__()
        self.n_joints = n_joints
        self.chunk = chunk
        self.encoder = CamEncoder(in_ch, img_hw, bottleneck)
        self.feat_dim = self.encoder.feat_dim
        h1, h2 = hidden
        head_in = self.feat_dim + n_joints
        self.head = nn.Sequential(
            nn.Linear(head_in, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, chunk * n_joints),
        )

    def forward(self, img, joints):
        f = self.encoder(img)
        out = self.head(torch.cat([f, joints], dim=1))
        return out.view(-1, self.chunk, self.n_joints)


def build_indices(starts, lengths, val_frac, rng, chunk_size, limit_episodes=None):
    E = len(lengths)
    if limit_episodes is not None:
        E = min(E, limit_episodes)
    ep_ids = np.arange(E)
    rng.shuffle(ep_ids)
    n_val = int(round(val_frac * E)) if E >= 2 else 0
    val_eps = set(ep_ids[:n_val].tolist())

    train_idx, val_idx = [], []
    for e in range(E):
        s, n = int(starts[e]), int(lengths[e])
        if n < 1 + chunk_size:
            continue
        frames = np.arange(s, s + n - chunk_size)
        (val_idx if e in val_eps else train_idx).append(frames)
    train_idx = np.concatenate(train_idx) if train_idx else np.empty(0, np.int64)
    val_idx = np.concatenate(val_idx) if val_idx else np.empty(0, np.int64)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, augment=False):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    use_amp = scaler is not None and device.type == "cuda"
    for s, jin, act in loader:
        s = s.to(device, non_blocking=True)
        jin = jin.to(device, non_blocking=True)
        act = act.to(device, non_blocking=True)
        if train and augment:
            s = gpu_augment(s)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(s, jin)
                loss = loss_fn(pred, act)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        bs = s.size(0)
        total += loss.item() * bs
        count += bs
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser(description="Action-chunked single-cam BC training.")
    ap.add_argument("--data", default="dataset_cube_stack.npz")
    ap.add_argument("--out", default="bc_cube_stack.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--ensemble-decay", type=float, default=0.01)
    ap.add_argument("--eval-max-steps", type=int, default=170)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device} | chunk={args.chunk}", flush=True)

    meta = np.load(args.data)
    joint_angles = meta["joint_angles"].astype(np.float32)
    starts = meta["episode_starts"]
    lengths = meta["episode_lengths"]
    joint_names = [str(s) for s in meta["joint_names"]]
    n_joints = joint_angles.shape[1]

    side_mm = npz_member_memmap(args.data, "images_side.npy")
    print(f"side images: {side_mm.shape} {side_mm.dtype} ({side_mm.nbytes/1e9:.1f} GB)",
          flush=True)

    import psutil
    avail_gb = psutil.virtual_memory().available / 1e9
    bps_side = int(np.prod(side_mm.shape[1:]))
    total_gb = side_mm.nbytes / 1e9
    headroom_gb = 12.0

    if total_gb + headroom_gb < avail_gb:
        load_frames = int(side_mm.shape[0])
        why = "full dataset fits in RAM"
    else:
        budget = (avail_gb - headroom_gb) * 1e9
        fit_frames = int(budget / bps_side)
        cum = np.cumsum(lengths)
        E_fit = int(np.searchsorted(cum, fit_frames, side="right"))
        load_frames = int(cum[E_fit - 1]) if E_fit > 0 else 0
        why = (f"auto-subsample: total {total_gb:.1f} GB > avail {avail_gb:.1f} GB; "
               f"loading {E_fit}/{len(lengths)} eps ({load_frames} frames)")
    print(f"RAM plan: {why}", flush=True)
    t_load = time.perf_counter()
    side = np.array(side_mm[:load_frames], copy=True)
    del side_mm
    print(f"loaded side {side.shape} in {time.perf_counter()-t_load:.1f}s", flush=True)

    cum = np.cumsum(lengths)
    n_eps_loaded = int(np.searchsorted(cum, load_frames, side="right"))
    starts = starts[:n_eps_loaded]
    lengths = lengths[:n_eps_loaded]
    print(f"episodes loaded: {n_eps_loaded}  frames loaded: {load_frames}", flush=True)

    train_idx, val_idx = build_indices(starts, lengths, args.val_frac, rng,
                                       args.chunk, args.limit_episodes)
    if len(train_idx) == 0:
        raise SystemExit("no training samples")

    tr_j = joint_angles[train_idx]
    j_mean = tr_j.mean(axis=0)
    j_std = tr_j.std(axis=0)
    j_std[j_std < 1e-6] = 1.0

    def norm_single(a):
        return ((joint_angles[a] - j_mean) / j_std).astype(np.float32)

    chunk_offsets = np.arange(1, args.chunk + 1)

    def chunk_targets(idx):
        full = idx[:, None] + chunk_offsets[None, :]
        return ((joint_angles[full] - j_mean) / j_std).astype(np.float32)

    train_ds = RAMSideChunkDataset(side, train_idx,
                                   norm_single(train_idx), chunk_targets(train_idx))
    val_ds = (RAMSideChunkDataset(side, val_idx,
                                  norm_single(val_idx), chunk_targets(val_idx))
              if len(val_idx) else None)
    print(f"samples: train={len(train_idx)}  val={len(val_idx)}  chunk={args.chunk}",
          flush=True)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              drop_last=True)
    val_loader = (DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)
                  if val_ds is not None else None)

    img_hw = tuple(side.shape[1:3])
    model = BCPolicySideChunk(n_joints=n_joints, in_ch=side.shape[3],
                              chunk=args.chunk, img_hw=img_hw,
                              bottleneck=args.bottleneck).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs, T_mult=1, eta_min=1e-7)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params, bottleneck={args.bottleneck}, "
          f"chunk={args.chunk}, img_hw={img_hw}", flush=True)

    def _save(path, epoch, train_loss, val_loss):
        torch.save({
            "model_state": model.state_dict(),
            "arch": {"n_joints": n_joints, "in_ch": int(side.shape[3]),
                     "chunk": args.chunk, "img_hw": img_hw,
                     "bottleneck": args.bottleneck,
                     "hidden": (512, 256)},
            "joint_mean": j_mean, "joint_std": j_std,
            "joint_names": joint_names,
            "action": "k future absolute joint targets (standardized), k=chunk",
            "normalization": "inputs/targets standardized; images /255",
            "epoch": epoch, "val_loss": val_loss, "train_loss": train_loss,
            "args": vars(args),
        }, path)

    pending_evals = []
    eval_log_path = args.out.replace(".pt", "_inflight_eval.log")
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = run_epoch(model, train_loader, loss_fn, device,
                               optimizer=optimizer, scaler=scaler,
                               augment=args.augment)
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(model, val_loader, loss_fn, device)
            monitor = val_loss
        else:
            val_loss, monitor = float("nan"), train_loss
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        improved = monitor < best
        if improved:
            best = monitor
            _save(args.out, epoch, train_loss, val_loss)
        dt = time.perf_counter() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.5f}  "
              f"val {val_loss:.5f}  lr {lr_now:.2e}  {dt:5.1f}s"
              f"{'  <- best, saved' if improved else ''}", flush=True)

        if epoch % 5 == 0 or epoch == args.epochs:
            snap = args.out.replace(".pt", f"_epoch{epoch}.pt")
            _save(snap, epoch, train_loss, val_loss)
            json_p = args.out.replace(".pt", f"_eval100_epoch{epoch}.json")
            log_p = args.out.replace(".pt", f"_eval100_epoch{epoch}.log")
            cmd = ["python3", "eval_bc_cube_stack.py", "--policy", snap,
                   "--episodes", "100", "--start-seed", "20000",
                   "--workers", "8", "--rate", "50",
                   "--max-steps", str(args.eval_max_steps),
                   "--ensemble-decay", str(args.ensemble_decay),
                   "--skip-solver",
                   "--save-json", json_p]
            env = os.environ.copy()
            for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
                      "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
                env[v] = "1"
            p = subprocess.Popen(cmd, stdout=open(log_p, "w"),
                                 stderr=subprocess.STDOUT, env=env)
            pending_evals.append((epoch, p, json_p))
            print(f"  [eval epoch {epoch}] launched bg subprocess pid={p.pid}",
                  flush=True)

        still = []
        for ep_done, proc, jp in pending_evals:
            if proc.poll() is None:
                still.append((ep_done, proc, jp))
            else:
                try:
                    import json
                    with open(jp) as f:
                        d = json.load(f)
                    ps = d.get("policy_summary") or {}
                    n_total = d.get("episodes", 1)
                    succ = ps.get("overall_success", 0)
                    pct = 100.0 * succ / max(n_total, 1)
                    sel = ps.get("correct_selection", "?")
                    gr = ps.get("grasp_success", "?")
                    pot = ps.get("place_on_top", "?")
                    line = (f"  [eval epoch {ep_done}] "
                            f"{succ}/{n_total} = {pct:.1f}%  "
                            f"(sel={sel} grasp={gr} place_on_top={pot})")
                    print(line, flush=True)
                    with open(eval_log_path, "a") as f:
                        f.write(line + "\n")
                except Exception as e:
                    print(f"  [eval epoch {ep_done}] no json ({e})", flush=True)
        pending_evals = still

    for ep_done, proc, jp in pending_evals:
        proc.wait()
        try:
            import json
            with open(jp) as f:
                d = json.load(f)
            ps = d.get("policy_summary") or {}
            n_total = d.get("episodes", 1)
            succ = ps.get("overall_success", 0)
            pct = 100.0 * succ / max(n_total, 1)
            line = f"  [eval epoch {ep_done}] {succ}/{n_total} = {pct:.1f}%"
            print(line, flush=True)
            with open(eval_log_path, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"  [eval epoch {ep_done}] no json ({e})", flush=True)

    print(f"\nDone. Best {'val' if val_loader else 'train'} {best:.5f} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
