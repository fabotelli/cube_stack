"""
check_pixel_floor.py — pixel-floor preflight for the two-IDENTICAL-RED-cube
stacking task.  Render N random scenes at 256x256 from policy_cam and measure:

  (a) the bounding-box pixel size of EACH cube in a crop around its projected
      centre (must clear the ~12 px minimum), and
  (b) DISTINGUISHABILITY: the two cubes must read as two SEPARATE red blobs, not
      a fused one.  We test this on the random scenes AND on forced worst-case
      pairs placed at exactly the spawn min-separation (0.045 m) at several
      orientations relative to the camera.  Two cubes are "separated" if, on the
      pixel segment joining their projected centres, there is at least one clearly
      non-red (background) pixel between the two red clusters.

Pass criteria:
  * every cube bbox max side >= 12 px,
  * mean saturation of detected cube pixels >= 0.45 (saturated red, not aliased),
  * every tested pair (random + forced-min-sep) is separable (a background gap
    exists between the two red clusters).

If the floor is not met, this prints FAIL and the offending measurements; the
caller should TIGHTEN FRAMING (move/zoom the camera) -- do NOT drop resolution.
"""

import os
import sys
import json
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import mujoco
import imageio.v2 as imageio

from cube_stack_env import CubeStackEnv, CUBE_MIN_SEP
from collect_dataset_cube_stack import _dr_setup, _dr_apply


RES = 256
N_SAMPLES = 8


def rgb_to_hsv(rgb01):
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    cmax = np.max(rgb01, axis=-1)
    cmin = np.min(rgb01, axis=-1)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta > 1e-9
    rmask = (cmax == r) & mask
    gmask = (cmax == g) & mask
    bmask = (cmax == b) & mask
    h[rmask] = (60 * ((g[rmask] - b[rmask]) / delta[rmask]) + 360) % 360
    h[gmask] = (60 * ((b[gmask] - r[gmask]) / delta[gmask]) + 120)
    h[bmask] = (60 * ((r[bmask] - g[bmask]) / delta[bmask]) + 240)
    s = np.where(cmax > 1e-9, delta / np.maximum(cmax, 1e-9), 0.0)
    v = cmax
    return h, s, v


def red_mask(rgb01):
    h, s, v = rgb_to_hsv(rgb01)
    return ((h <= 25) | (h >= 335)) & (s >= 0.45) & (v >= 0.20)


def circular_mean_deg(h_deg):
    rad = np.deg2rad(np.asarray(h_deg, dtype=np.float64))
    a = np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
    return float(a % 360)


def project_point(model, data, point_world, cam_name, res):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    cam_local = cam_mat.T @ (point_world - cam_pos)
    z = -cam_local[2]
    if z <= 1e-4:
        return None
    fovy_rad = np.deg2rad(float(model.cam_fovy[cam_id]))
    f = (res / 2.0) / np.tan(fovy_rad / 2.0)
    col = int(round(res / 2.0 + cam_local[0] / z * f))
    row = int(round(res / 2.0 - cam_local[1] / z * f))
    if 0 <= row < res and 0 <= col < res:
        return row, col
    return None


def measure_cube(frame, model, data, body, res, *, crop_half=20):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    projected = project_point(model, data, data.xpos[body_id].copy(),
                              "policy_cam", res)
    if projected is None:
        return None
    r0, c0 = projected
    rlo, rhi = max(0, r0 - crop_half), min(res, r0 + crop_half + 1)
    clo, chi = max(0, c0 - crop_half), min(res, c0 + crop_half + 1)
    crop = frame[rlo:rhi, clo:chi].astype(np.float32) / 255.0
    h, s, v = rgb_to_hsv(crop)
    mask = ((h <= 25) | (h >= 335)) & (s >= 0.45) & (v >= 0.20)
    if not mask.any():
        return dict(centre_rc=[r0, c0], pix_count=0, bbox_hw=[0, 0],
                    hue_circ_mean=None, sat_mean=None)
    ys, xs = np.where(mask)
    return dict(centre_rc=[r0, c0], pix_count=int(mask.sum()),
                bbox_hw=[int(ys.max() - ys.min() + 1),
                         int(xs.max() - xs.min() + 1)],
                hue_circ_mean=circular_mean_deg(h[mask]),
                sat_mean=float(s[mask].mean()),
                val_mean=float(v[mask].mean()))


def separable(frame, model, data, res):
    """Is there a non-red gap between the two cubes' projected centres?"""
    pa = project_point(model, data, data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_a")].copy(),
        "policy_cam", res)
    pb = project_point(model, data, data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_b")].copy(),
        "policy_cam", res)
    if pa is None or pb is None:
        return None
    rm = red_mask(frame.astype(np.float32) / 255.0)
    n = max(abs(pa[0] - pb[0]), abs(pa[1] - pb[1])) + 1
    rows = np.linspace(pa[0], pb[0], n).round().astype(int)
    cols = np.linspace(pa[1], pb[1], n).round().astype(int)
    seg = rm[rows, cols]
    centre_px_dist = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
    # count the longest run of consecutive non-red pixels along the segment
    gap = 0
    best = 0
    for val in seg:
        if not val:
            gap += 1
            best = max(best, gap)
        else:
            gap = 0
    return dict(centre_px_dist=centre_px_dist, max_bg_gap=int(best),
                separated=bool(best >= 1))


def _set_cube(env, c, x, y, yaw):
    q = env.cube_qadr[c]
    env.data.qpos[q:q + 3] = [x, y, env.table_top_z + env.cube_half + 1e-3]
    env.data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    env.cube_xy0[c] = np.array([x, y])


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    env = CubeStackEnv()
    env.model.vis.global_.offwidth = RES
    env.model.vis.global_.offheight = RES
    renderer = mujoco.Renderer(env.model, height=RES, width=RES)
    dr = _dr_setup(env.model)

    report = {"resolution": RES, "samples": [], "forced_min_sep": []}
    sizes, sats, hues = [], [], []
    sep_fail = 0

    # --- random scenes ---------------------------------------------------- #
    for i in range(N_SAMPLES):
        seed = 9000 + i
        env.reset(seed=seed)
        _dr_apply(env.model, np.random.default_rng(seed), dr)
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera="policy_cam")
        frame = renderer.render()
        imageio.imwrite(os.path.join(out_dir, f"pixel_floor_sample_{i:02d}.png"),
                        frame)
        ca = measure_cube(frame, env.model, env.data, "cube_a", RES)
        cb = measure_cube(frame, env.model, env.data, "cube_b", RES)
        sep = separable(frame, env.model, env.data, RES)
        for m in (ca, cb):
            if m and m["pix_count"]:
                sizes.append(max(m["bbox_hw"]))
                sats.append(m["sat_mean"])
                hues.append(m["hue_circ_mean"])
            else:
                sizes.append(0)
        if sep and not sep["separated"]:
            sep_fail += 1
        report["samples"].append(dict(seed=seed, cube_a=ca, cube_b=cb, sep=sep,
                                       far_near=env.pick_order()))
        am = max(ca["bbox_hw"]) if ca and ca["pix_count"] else 0
        bm = max(cb["bbox_hw"]) if cb and cb["pix_count"] else 0
        print(f"seed {seed} | a bbox_max={am}px b bbox_max={bm}px | "
              f"centre_dist={sep['centre_px_dist']:.1f}px gap={sep['max_bg_gap']}px "
              f"sep={sep['separated']}")

    # --- forced worst-case min-separation pairs --------------------------- #
    # Place the pair at exactly the 0.045 m spawn min-sep, centred in the cube
    # zone, at several orientations of the separation vector relative to camera.
    min_sep = CUBE_MIN_SEP
    cx, cy = 0.157, -0.020
    for k, ang in enumerate(np.linspace(0, np.pi, 6, endpoint=False)):
        env.reset(seed=12000 + k)
        dx, dy = (min_sep / 2) * np.cos(ang), (min_sep / 2) * np.sin(ang)
        _set_cube(env, "a", cx - dx, cy - dy, 0.3)
        _set_cube(env, "b", cx + dx, cy + dy, 1.1)
        env.data.qvel[:] = 0.0
        _dr_apply(env.model, np.random.default_rng(12000 + k), dr)
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera="policy_cam")
        frame = renderer.render()
        imageio.imwrite(os.path.join(out_dir, f"min_sep_{k:02d}.png"), frame)
        ca = measure_cube(frame, env.model, env.data, "cube_a", RES)
        cb = measure_cube(frame, env.model, env.data, "cube_b", RES)
        sep = separable(frame, env.model, env.data, RES)
        if sep and not sep["separated"]:
            sep_fail += 1
        am = max(ca["bbox_hw"]) if ca and ca["pix_count"] else 0
        bm = max(cb["bbox_hw"]) if cb and cb["pix_count"] else 0
        report["forced_min_sep"].append(dict(angle_deg=float(np.rad2deg(ang)),
                                              cube_a=ca, cube_b=cb, sep=sep))
        print(f"min_sep ang={np.rad2deg(ang):5.1f} | a={am}px b={bm}px | "
              f"centre_dist={sep['centre_px_dist']:.1f}px gap={sep['max_bg_gap']}px "
              f"sep={sep['separated']}")

    renderer.close()

    sizes = np.asarray(sizes, float)
    pass_floor = bool((sizes >= 12).all())
    sat_ok = bool(np.mean(sats) >= 0.45) if sats else False
    sep_ok = sep_fail == 0
    report["summary"] = dict(
        cube_bbox_max_side=dict(min=float(sizes.min()), max=float(sizes.max()),
                                mean=float(sizes.mean())),
        cube_sat_mean=float(np.mean(sats)) if sats else None,
        cube_hue_circ_mean=circular_mean_deg(hues) if hues else None,
        n_separation_failures=int(sep_fail),
        pass_pixel_floor=pass_floor,
        pass_distinguishable=sep_ok,
        pass_saturation=sat_ok,
    )
    with open(os.path.join(out_dir, "pixel_floor_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nSummary:")
    print(json.dumps(report["summary"], indent=2))
    ok = pass_floor and sep_ok and sat_ok
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "pixel_floor_check"
    sys.exit(main(out))
