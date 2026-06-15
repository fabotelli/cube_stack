"""
cube_stack_solver.py
====================

Privileged ("teacher") FSM for the two-cube STACKING task.  Same pick-and-place
primitives as ``dice_sort_solver`` / ``privileged_solver``, but a SINGLE
pick-and-place per episode:

    pick the cube FARTHER from gripper home  ->  place it ON TOP of the cube
    NEARER to gripper home.

Sequencing rule is GEOMETRY-ONLY: the farther-from-home cube is the one that
gets lifted (a tie is broken canonical 'a','b').  The cubes are identical, so
the policy never has an appearance cue -- it must infer the same farther/nearer
decision from the side camera alone.

Precise placement: after the lift, we read the privileged offset between the
held cube's centre and the pinch site, then command the pinch so the *cube*
(not the pinch) is centred a hair above the base cube's top face.  This keeps
the placement inside the tight ~cube-width tolerance.

GRIP and MOVE are separate steps (never overlap a gripper command with a
Cartesian move).
"""

from __future__ import annotations

import time
import numpy as np
import mujoco

from cube_stack_env import (CubeStackEnv, APPROACH_DOWN,
                            GRIPPER_OPEN, GRIPPER_CLOSE, CUBES)


# Waypoint offsets (mirror dice_sort_solver).
STANDOFF_HEIGHT = 0.05    # standoff above cube pre-/post-grasp
GRASP_DROP = 0.014        # grasp this far below cube centre (low grip)

# Stack-side waypoints.  Carry the held cube high over the base cube, then lower
# so the held cube's bottom face rests a few mm above the base cube's top.
STACK_STANDOFF_HEIGHT = 0.085   # cube centre carried this far above base centre
STACK_RELEASE_GAP = 0.010       # release with cube bottom this far above base top
                                # (keeps the fingertips clear of the base so the
                                #  descent doesn't shove it before release)


class CubeStackSolver:
    """Single pick-and-place FSM: pick farther cube, stack on nearer cube."""

    def __init__(self, env: CubeStackEnv, speed: float | None = None,
                 recorder=None):
        self.env = env
        self._recorder = recorder
        self._step_sleep = (None if speed is None
                            else env.model.opt.timestep / speed)

    def _render(self, viewer) -> None:
        if self._recorder is not None:
            self._recorder.capture(self.env.data)
        if viewer is None:
            return
        viewer.sync()
        if self._step_sleep:
            time.sleep(self._step_sleep)

    # ------------------------------------------------------------------ #
    #  Motion primitives (verbatim from dice_sort_solver).
    # ------------------------------------------------------------------ #
    def _move_to_pose(self, target_pos, *,
                      gripper, pin_wrist_roll=None, settle_tol=0.02,
                      max_steps=1500, viewer=None):
        env = self.env
        q_target = env.solve_ik(target_pos, APPROACH_DOWN,
                                q_init=env.arm_qpos_now,
                                pin_wrist_roll=pin_wrist_roll)
        env.set_gripper(gripper)
        slew = 0.005
        q_cmd = env.arm_qpos_now
        for _ in range(max_steps):
            step = np.clip(q_target - q_cmd, -slew, slew)
            q_cmd = q_cmd + step
            env.set_arm_target(q_cmd)
            env.step()
            self._render(viewer)
            if np.max(np.abs(env.arm_qpos_now - q_target)) < settle_tol:
                return True
        return False

    def _hold(self, steps, *, gripper, viewer=None):
        env = self.env
        env.set_gripper(gripper)
        for _ in range(steps):
            env.step()
            self._render(viewer)

    def _close_grip(self, *, ramp=250, hold=200, viewer=None):
        env = self.env
        for i in range(ramp):
            env.set_gripper(GRIPPER_OPEN + (GRIPPER_CLOSE - GRIPPER_OPEN)
                            * (i + 1) / ramp)
            env.step()
            self._render(viewer)
        for _ in range(hold):
            env.step()
            self._render(viewer)

    # ------------------------------------------------------------------ #
    #  Per-cube wrist-roll alignment (probe wr=0, align jaws to cube yaw).
    # ------------------------------------------------------------------ #
    def _wrist_roll_for_cube(self, c: str) -> float:
        env = self.env
        cube_xyz = env.cube_pos(c)
        cube_q = env.cube_quat(c)
        cube_yaw = 2.0 * np.arctan2(cube_q[3], cube_q[0])
        q_probe = env.solve_ik(cube_xyz + np.array([0, 0, STANDOFF_HEIGHT]),
                               q_init=env.home_seed, pin_wrist_roll=0.0)
        env._ik_data.qpos[env.arm_qpos] = q_probe
        mujoco.mj_fwdPosition(env.model, env._ik_data)
        pmat = env._ik_data.site_xmat[env.site_pinch].reshape(3, 3)
        jaw_at_wr0 = float(np.arctan2(pmat[1, 1], pmat[0, 1]))
        return ((jaw_at_wr0 - cube_yaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4

    # ------------------------------------------------------------------ #
    #  Whole-episode FSM: pick farther cube, stack onto nearer cube.
    # ------------------------------------------------------------------ #
    def run_episode(self, viewer=None) -> dict:
        env = self.env
        farther, nearer = env.pick_order()

        pick_xyz = env.cube_pos(farther)
        grasp_xyz = pick_xyz - np.array([0, 0, GRASP_DROP])
        standoff_xyz = pick_xyz + np.array([0, 0, STANDOFF_HEIGHT])

        wr = self._wrist_roll_for_cube(farther)
        log = {"farther": farther, "nearer": nearer}

        # 1) APPROACH: standoff above the farther cube, jaws open.
        log["approach"] = self._move_to_pose(
            standoff_xyz, gripper=GRIPPER_OPEN, pin_wrist_roll=wr, viewer=viewer)

        # 2) DESCEND: lower onto the cube.
        log["descend"] = self._move_to_pose(
            grasp_xyz, gripper=GRIPPER_OPEN, pin_wrist_roll=wr,
            settle_tol=0.01, viewer=viewer)

        # 3) GRASP: close (separate step from the move).
        self._close_grip(viewer=viewer)

        # 4) LIFT: back to standoff, jaws closed.
        log["lift"] = self._move_to_pose(
            standoff_xyz, gripper=GRIPPER_CLOSE, pin_wrist_roll=wr, viewer=viewer)

        # Privileged grasp offset: where the held cube sits relative to the
        # pinch right now.  We use it so the CUBE (not the pinch) lands centred
        # over the base cube's top -- needed for the tight stack tolerance.
        offset = env.cube_pos(farther) - env.pinch_pos

        base_xyz = env.cube_pos(nearer)
        base_top_z = base_xyz[2] + env.cube_half
        # Desired held-cube centre when released: bottom face a small gap above
        # the base top.
        place_cube_centre = np.array([
            base_xyz[0], base_xyz[1],
            base_top_z + env.cube_half + STACK_RELEASE_GAP])
        stand_cube_centre = np.array([
            base_xyz[0], base_xyz[1], base_xyz[2] + STACK_STANDOFF_HEIGHT])

        pinch_place = place_cube_centre - offset
        pinch_stand = stand_cube_centre - offset

        # 5) OVER BASE: hover the held cube above the base cube.
        log["over_base"] = self._move_to_pose(
            pinch_stand, gripper=GRIPPER_CLOSE, pin_wrist_roll=wr, viewer=viewer)

        # 6) LOWER: descend so the held cube is just above the base top.
        log["lower"] = self._move_to_pose(
            pinch_place, gripper=GRIPPER_CLOSE, pin_wrist_roll=wr,
            settle_tol=0.01, viewer=viewer)

        # 7) RELEASE: open and let the cube settle onto the base.
        self._hold(400, gripper=GRIPPER_OPEN, viewer=viewer)

        # 8) RETREAT: rise straight up so the gripper doesn't topple the stack.
        log["retreat"] = self._move_to_pose(
            pinch_stand, gripper=GRIPPER_OPEN, pin_wrist_roll=wr, viewer=viewer)

        success = env.is_stacked()
        return dict(
            farther=farther, nearer=nearer, phase=log,
            success=bool(success),
            cube_a_final=env.cube_pos("a"),
            cube_b_final=env.cube_pos("b"),
        )


# --------------------------------------------------------------------------- #
#  Headless smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    env = CubeStackEnv()
    if args.viewer:
        import mujoco.viewer
        env.reset(seed=args.seed)
        solver = CubeStackSolver(env, speed=1.0)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.sync()
            log = solver.run_episode(viewer=viewer)
            print(log)
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
    else:
        solver = CubeStackSolver(env)
        n_ok = 0
        for i in range(args.episodes):
            env.reset(seed=args.seed + i)
            t0 = time.perf_counter()
            log = solver.run_episode()
            dt = time.perf_counter() - t0
            n_ok += int(log["success"])
            print(f"ep {i:3d} farther={log['farther']} nearer={log['nearer']} "
                  f"success={log['success']}  {dt:.1f}s")
        n = args.episodes
        print(f"\n{n_ok}/{n} stacked = {100*n_ok/n:.1f}%")
