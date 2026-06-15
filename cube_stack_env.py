"""
cube_stack_env.py
=================

Wrapper around ``learm_scene_cube_stack.xml`` for the two-cube STACKING task.
Same 5-DOF LeArm + 6-joint IK as ``pick_place_env.PickPlaceEnv`` /
``dice_sort_env.DiceSortEnv``, but:

  * Two IDENTICAL red cubes (``cube_a``, ``cube_b``), both free joints, both
    randomised per reset (positions + yaws), non-overlapping, both in reach.
  * NO bins.  The goal is to STACK the FARTHER cube on top of the NEARER cube,
    where "farther / nearer" is Euclidean distance from the gripper HOME xy in
    the workspace plane (privileged, computed from ground truth).
  * Per-cube privileged accessors plus a helper returning the pick order
    (FARTHER first -- it gets picked up and placed on the NEARER one).
  * Privileged success check: the farther cube is "stacked" on the nearer cube
    if its xy is within STACK_XY_TOL of the nearer cube's xy AND it sits a full
    cube-height above it (its centre clearly higher), with the nearer cube
    still resting on the table (the base wasn't knocked away / lifted).

The student policy only ever sees the side camera + joint angles.  Everything
exposed here is privileged state used by the teacher solver / eval metrics.
"""

from __future__ import annotations

import os
import numpy as np
import mujoco


ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll"]
ARM_ACTS = ["a_pan", "a_lift", "a_elbow", "a_wflex", "a_wroll"]
GRIP_ACT = "a_grip"

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = -1.5
APPROACH_DOWN = np.array([0.0, 0.0, -1.0])

CUBES = ("a", "b")  # the two identical cubes -- distinguished only by geometry

# Stack success: farther-cube centre within this radius (xy) of the nearer-cube
# centre after release.  ~one cube width (cube side = 1.9 cm) -> tight placement.
STACK_XY_TOL = 0.018
# The stacked (top) cube centre must sit at least this much above the base cube
# centre (a clean stack is ~one full side = 0.019 m higher).
STACK_DZ_MIN = 0.012

# Minimum centre-to-centre spawn separation.  Set so the two identical cubes
# stay separable as two red blobs in the side camera even in the worst
# camera-depth-aligned orientation (verified by check_pixel_floor: worst-case
# background gap ~6 px at 0.065; they fuse below ~0.055).
CUBE_MIN_SEP = 0.065


class CubeStackEnv:
    """5-DOF LeArm two-cube stacking environment with privileged state access."""

    def __init__(self, xml_path: str | None = None):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "learm_scene_cube_stack.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)

        m = self.model

        def jid(name):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)

        self.arm_qpos = np.array([m.jnt_qposadr[jid(j)] for j in ARM_JOINTS])
        self.arm_dofs = np.array([m.jnt_dofadr[jid(j)] for j in ARM_JOINTS])
        self.arm_range = np.array([m.jnt_range[jid(j)] for j in ARM_JOINTS])

        self.arm_act = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
             for a in ARM_ACTS])
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIP_ACT)

        self.site_pinch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")

        # Two cubes (free joints).
        self.body_cube = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"cube_{c}")
                          for c in CUBES}
        self.cube_qadr = {c: m.jnt_qposadr[jid(f"cube_{c}_free")] for c in CUBES}

        self.home_seed = np.array([0.0, 1.1, -1.3, 0.6, 0.0])

        # Pick-zone home (where the gripper starts each episode).  Distance from
        # this point decides farther vs nearer -- a geometry rule, independent
        # of appearance, that the student must infer from pixels alone.
        self.home_xy = np.array([0.145, 0.0])

        # Cube size: half-side of the cube geom.
        self.cube_half = float(m.geom_size[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_a_geom")][2])
        self.table_top_z = 0.15

        self._rng = np.random.default_rng()

        # Cached per-episode privileged spawn state (eval reads it; never the
        # student).
        self.cube_xy0 = {c: np.zeros(2) for c in CUBES}

    # --------------------------------------------------------------------- #
    #  Reset / randomization
    # --------------------------------------------------------------------- #
    def _sample_xy_separated(self, x_lo, x_hi, y_lo, y_hi, min_sep, others=()):
        """Rejection-sample one (x, y) at least `min_sep` from every other xy."""
        for _ in range(200):
            x = self._rng.uniform(x_lo, x_hi)
            y = self._rng.uniform(y_lo, y_hi)
            ok = True
            for ox, oy in others:
                if np.hypot(x - ox, y - oy) < min_sep:
                    ok = False
                    break
            if ok:
                return x, y
        return x_hi, y_hi

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Home pose (same as PickPlaceEnv).
        home_q = self.solve_ik(np.array([0.145, 0.0, 0.21]),
                               q_init=self.home_seed)
        self.data.qpos[self.arm_qpos] = home_q
        self.data.ctrl[self.arm_act] = home_q
        self.data.ctrl[self.grip_act] = GRIPPER_OPEN

        # --- Cube placements ---------------------------------------------- #
        # Both cubes in the top-down-IK-reachable patch.  CUBE_MIN_SEP keeps
        # them (a) graspable one-at-a-time without the gripper fouling the
        # other, and (b) two distinct red blobs in the side camera (verified by
        # the pixel-floor preflight at exactly min_sep).
        cube_xy = {}
        cube_yaw = {}
        for c in CUBES:
            x, y = self._sample_xy_separated(
                0.130, 0.185, -0.085, 0.045, min_sep=CUBE_MIN_SEP,
                others=list(cube_xy.values()))
            cube_xy[c] = (x, y)
            cube_yaw[c] = float(self._rng.uniform(-np.pi, np.pi))

        # --- Write to physics --------------------------------------------- #
        cz = self.table_top_z + self.cube_half + 1e-3
        for c in CUBES:
            q = self.cube_qadr[c]
            x, y = cube_xy[c]
            yaw = cube_yaw[c]
            self.data.qpos[q:q + 3] = [x, y, cz]
            self.data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.cube_xy0[c] = np.array([x, y])

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------------------- #
    #  Privileged accessors
    # --------------------------------------------------------------------- #
    def cube_pos(self, c: str) -> np.ndarray:
        return self.data.xpos[self.body_cube[c]].copy()

    def cube_quat(self, c: str) -> np.ndarray:
        q = self.cube_qadr[c]
        return self.data.qpos[q + 3:q + 7].copy()

    @property
    def pinch_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site_pinch].copy()

    @property
    def approach_axis(self) -> np.ndarray:
        return self.data.site_xmat[self.site_pinch].reshape(3, 3)[:, 2].copy()

    @property
    def arm_qpos_now(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos].copy()

    def dist_from_home(self, c: str) -> float:
        """Euclidean distance (workspace plane) from gripper home to the cube's
        SPAWN xy.  Spawn-based so the farther/nearer labels are fixed for the
        whole episode -- once a cube is lifted/stacked its live xy moves and
        would otherwise flip the order."""
        return float(np.linalg.norm(self.cube_xy0[c] - self.home_xy))

    def pick_order(self) -> tuple[str, str]:
        """Return (farther, nearer): the FARTHER-from-home cube is picked up and
        placed on top of the NEARER one.  Ties broken by canonical ('a','b').
        Geometry-only -- the student must infer the same order from pixels."""
        da, db = self.dist_from_home("a"), self.dist_from_home("b")
        if abs(da - db) < 1e-9:
            return ("a", "b")
        farther = "a" if da > db else "b"
        nearer = "b" if farther == "a" else "a"
        return (farther, nearer)

    def is_stacked(self) -> bool:
        """Privileged success check: the farther cube is stably stacked on top
        of the nearer cube."""
        farther, nearer = self.pick_order()
        pf = self.cube_pos(farther)
        pn = self.cube_pos(nearer)
        dxy = float(np.linalg.norm(pf[:2] - pn[:2]))
        dz = float(pf[2] - pn[2])
        base_on_table = pn[2] < self.table_top_z + self.cube_half + 0.012
        return bool(dxy < STACK_XY_TOL and dz > STACK_DZ_MIN and base_on_table)

    # --------------------------------------------------------------------- #
    #  IK (verbatim from PickPlaceEnv / DiceSortEnv)
    # --------------------------------------------------------------------- #
    def solve_ik(self,
                 target_pos: np.ndarray,
                 approach_dir: np.ndarray = APPROACH_DOWN,
                 q_init: np.ndarray | None = None,
                 pin_wrist_roll: float | None = None,
                 pos_tol: float = 8e-4,
                 rot_tol: float = 3e-2,
                 max_iters: int = 300,
                 damping: float = 0.05) -> np.ndarray:
        m, d = self.model, self._ik_data
        approach_dir = approach_dir / np.linalg.norm(approach_dir)

        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0.0
        if q_init is not None:
            d.qpos[self.arm_qpos] = q_init

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        err = np.zeros(6)

        for _ in range(max_iters):
            mujoco.mj_fwdPosition(m, d)

            err[:3] = target_pos - d.site_xpos[self.site_pinch]
            axis = d.site_xmat[self.site_pinch].reshape(3, 3)[:, 2]
            err[3:] = np.cross(axis, approach_dir)

            if (np.linalg.norm(err[:3]) < pos_tol and
                    np.linalg.norm(err[3:]) < rot_tol):
                break

            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_pinch)
            J = np.vstack([jacp[:, self.arm_dofs], jacr[:, self.arm_dofs]])

            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(6), err)

            q = d.qpos[self.arm_qpos] + dq
            q = np.clip(q, self.arm_range[:, 0], self.arm_range[:, 1])
            d.qpos[self.arm_qpos] = q

        if pin_wrist_roll is not None:
            d.qpos[self.arm_qpos[4]] = np.clip(
                pin_wrist_roll, self.arm_range[4, 0], self.arm_range[4, 1])

        return d.qpos[self.arm_qpos].copy()

    # --------------------------------------------------------------------- #
    #  Low-level command helpers
    # --------------------------------------------------------------------- #
    def set_arm_target(self, q: np.ndarray) -> None:
        self.data.ctrl[self.arm_act] = q

    def set_gripper(self, opening: float) -> None:
        self.data.ctrl[self.grip_act] = opening

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
