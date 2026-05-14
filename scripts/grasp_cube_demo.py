from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tv_isaaclab import EpisodeRecorder, IsaacLabEnvBridge, add_app_launcher_args, launch_simulation_app
from tv_isaaclab.contracts import TELEOP_TASK_ID, assemble_teleop_action, expand_inspire_driver_qpos


HAND_QUAT_XYZW = np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float32)
LEFT_HOME = np.array([-0.34, 0.18, 1.35], dtype=np.float32)
RIGHT_HOME = np.array([-0.34, -0.18, 1.35], dtype=np.float32)
HEAD_RMAT = np.eye(3, dtype=np.float32)
CUBE_SIZE = 0.05


@dataclass(frozen=True)
class Phase:
    name: str
    steps: int
    left_start: np.ndarray
    left_end: np.ndarray
    right_start: np.ndarray
    right_end: np.ndarray
    left_grip_start: float
    left_grip_end: float
    right_grip_start: float
    right_grip_end: float


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


def _make_pose(position: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(position, dtype=np.float32), HAND_QUAT_XYZW], axis=0)


def _grip_to_driver_qpos(grip: float) -> np.ndarray:
    grip = float(np.clip(grip, 0.0, 1.0))
    open_driver = np.zeros(6, dtype=np.float32)
    closed_driver = np.array([0.75, 0.75, 0.72, 0.72, 0.15, 0.42], dtype=np.float32)
    return expand_inspire_driver_qpos(_lerp(open_driver, closed_driver, grip))


def _build_demo_phases(cube_center: np.ndarray) -> list[Phase]:
    # Build the trajectory from the cube's current center instead of hard-coding world targets.
    cube_top_z = float(cube_center[2] + 0.5 * CUBE_SIZE)
    wrist_x = float(cube_center[0] - 0.085)
    wrist_y = float(cube_center[1] + 0.005)
    wrist_grasp_z = cube_top_z + 0.060
    wrist_pregrasp_z = wrist_grasp_z + 0.080
    wrist_lift_z = wrist_grasp_z + 0.115
    grasp_offset = np.array([wrist_x - cube_center[0], wrist_y - cube_center[1], wrist_grasp_z - cube_center[2]], dtype=np.float32)
    pregrasp_offset = np.array([wrist_x - cube_center[0], wrist_y - cube_center[1], wrist_pregrasp_z - cube_center[2]], dtype=np.float32)
    lift_offset = np.array([wrist_x - cube_center[0], wrist_y - cube_center[1], wrist_lift_z - cube_center[2]], dtype=np.float32)
    left_pregrasp = cube_center + pregrasp_offset
    left_grasp = cube_center + grasp_offset
    left_lift = cube_center + lift_offset
    right_watch = np.array([-0.22, -0.16, 1.35], dtype=np.float32)
    return [
        Phase(
            name="settle",
            steps=20,
            left_start=LEFT_HOME,
            left_end=LEFT_HOME,
            right_start=RIGHT_HOME,
            right_end=RIGHT_HOME,
            left_grip_start=0.0,
            left_grip_end=0.0,
            right_grip_start=0.0,
            right_grip_end=0.0,
        ),
        Phase(
            name="align_above_cube",
            steps=55,
            left_start=LEFT_HOME,
            left_end=left_pregrasp,
            right_start=RIGHT_HOME,
            right_end=right_watch,
            left_grip_start=0.0,
            left_grip_end=0.0,
            right_grip_start=0.0,
            right_grip_end=0.0,
        ),
        Phase(
            name="vertical_descend",
            steps=40,
            left_start=left_pregrasp,
            left_end=left_grasp,
            right_start=right_watch,
            right_end=right_watch,
            left_grip_start=0.0,
            left_grip_end=0.0,
            right_grip_start=0.0,
            right_grip_end=0.0,
        ),
        Phase(
            name="close",
            steps=35,
            left_start=left_grasp,
            left_end=left_grasp,
            right_start=right_watch,
            right_end=right_watch,
            left_grip_start=0.0,
            left_grip_end=1.0,
            right_grip_start=0.0,
            right_grip_end=0.15,
        ),
        Phase(
            name="lift",
            steps=80,
            left_start=left_grasp,
            left_end=left_lift,
            right_start=right_watch,
            right_end=right_watch,
            left_grip_start=1.0,
            left_grip_end=1.0,
            right_grip_start=0.15,
            right_grip_end=0.15,
        ),
        Phase(
            name="hold",
            steps=90,
            left_start=left_lift,
            left_end=left_lift,
            right_start=right_watch,
            right_end=right_watch,
            left_grip_start=1.0,
            left_grip_end=1.0,
            right_grip_start=0.15,
            right_grip_end=0.15,
        ),
    ]


def _iter_actions(cube_center: np.ndarray):
    for phase in _build_demo_phases(cube_center):
        for step_idx in range(phase.steps):
            alpha = 1.0 if phase.steps <= 1 else step_idx / float(phase.steps - 1)
            left_pos = _lerp(phase.left_start, phase.left_end, alpha)
            right_pos = _lerp(phase.right_start, phase.right_end, alpha)
            left_grip = (1.0 - alpha) * phase.left_grip_start + alpha * phase.left_grip_end
            right_grip = (1.0 - alpha) * phase.right_grip_start + alpha * phase.right_grip_end
            yield (
                phase.name,
                _make_pose(left_pos),
                _make_pose(right_pos),
                _grip_to_driver_qpos(left_grip),
                _grip_to_driver_qpos(right_grip),
                left_grip,
            )


def _set_cube_pose(env: IsaacLabEnvBridge, position: np.ndarray) -> None:
    env_target = getattr(env, "_env_target", None)
    if env_target is None or not hasattr(env_target, "cube"):
        return
    cube_pose = np.zeros((1, 7), dtype=np.float32)
    cube_pose[0, :3] = position
    cube_pose[0, 3] = 1.0
    cube_pose_t = env_target.adapt_action(cube_pose)
    env_target.cube.write_root_pose_to_sim(cube_pose_t)
    env_target.cube.write_root_velocity_to_sim(env_target.adapt_action(np.zeros((1, 6), dtype=np.float32)))


def _try_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception:
        return None


def _current_cube_center(env: IsaacLabEnvBridge) -> np.ndarray:
    env_target = getattr(env, "_env_target", None)
    if env_target is None or not hasattr(env_target, "cube"):
        return np.array([0.0, 0.0, 1.25], dtype=np.float32)

    cube = env_target.cube
    cube_data = getattr(cube, "data", None)
    if cube_data is not None:
        for attr_name in ("root_pos_w", "root_state_w", "default_root_state"):
            raw = getattr(cube_data, attr_name, None)
            array = _try_numpy(raw)
            if array is None or array.size < 3:
                continue
            if array.ndim == 1:
                return array[:3].astype(np.float32)
            return array[0, :3].astype(np.float32)

    env_origins = _try_numpy(getattr(getattr(env_target, "scene", None), "env_origins", None))
    if env_origins is not None and env_origins.size >= 3:
        origin = env_origins[0] if env_origins.ndim > 1 else env_origins[:3]
        return origin[:3].astype(np.float32) + np.array([0.0, 0.0, 1.25], dtype=np.float32)
    return np.array([0.0, 0.0, 1.25], dtype=np.float32)


def _cube_attach_position(left_pose: np.ndarray) -> np.ndarray:
    # Keep the cube just below the wrist after closure without forcing it into the table.
    return left_pose[:3] + np.array([0.085, -0.005, -0.045], dtype=np.float32)


def _hold_final_pose(
    env: IsaacLabEnvBridge,
    simulation_app,
    *,
    duration_s: float,
    loop_hz: float,
    action: np.ndarray,
    left_pose: np.ndarray,
    left_grip: float,
    assist_cube: bool,
) -> None:
    if duration_s <= 0.0:
        return
    deadline = time.perf_counter() + duration_s
    target_dt = 1.0 / max(loop_hz, 1.0)
    print(f"[*] Holding lifted pose for {duration_s:.1f}s")
    while simulation_app.is_running() and time.perf_counter() < deadline:
        loop_start = time.perf_counter()
        env.step(action, head_rmat=HEAD_RMAT)
        if assist_cube and left_grip >= 0.8:
            _set_cube_pose(env, _cube_attach_position(left_pose))
        elapsed = time.perf_counter() - loop_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


def _stay_open_until_closed(
    env: IsaacLabEnvBridge,
    simulation_app,
    *,
    loop_hz: float,
    action: np.ndarray,
    left_pose: np.ndarray,
    left_grip: float,
    assist_cube: bool,
) -> None:
    target_dt = 1.0 / max(loop_hz, 1.0)
    print("[*] Demo finished. Keeping final pose until you close the window or press Ctrl+C.")
    try:
        while simulation_app.is_running():
            loop_start = time.perf_counter()
            env.step(action, head_rmat=HEAD_RMAT)
            if assist_cube and left_grip >= 0.8:
                _set_cube_pose(env, _cube_attach_position(left_pose))
            elapsed = time.perf_counter() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
    except KeyboardInterrupt:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Scripted cube grasp demo for the Isaac Lab teleop scene")
    parser.add_argument("--task", type=str, default=TELEOP_TASK_ID)
    parser.add_argument("--loop_hz", type=float, default=30.0)
    parser.add_argument(
        "--post_grasp_hold_s",
        type=float,
        default=5.0,
        help="Extra time to keep the lifted final pose visible before exiting.",
    )
    parser.add_argument(
        "--stay_open",
        action="store_true",
        help="After the scripted sequence finishes, keep the window alive until you close it.",
    )
    parser.add_argument(
        "--respect_app_running",
        action="store_true",
        help="Stop the scripted sequence as soon as SimulationApp reports not running. "
        "By default the demo continues its finite scripted steps and relies on step exceptions instead.",
    )
    parser.add_argument("--record", action="store_true", help="Record the scripted demo to an HDF5 episode")
    parser.add_argument(
        "--output",
        type=str,
        default="../data/demos/grasp_cube_demo/processed_episode_0.hdf5",
        help="Episode output path when --record is enabled",
    )
    parser.add_argument(
        "--allow_fallback",
        action="store_true",
        help="Allow running against the non-USD fallback adapter instead of requiring the real Isaac Lab scene.",
    )
    parser.add_argument(
        "--assist_cube",
        action="store_true",
        help="After the fingers close, kinematically keep the cube in the grasp for a guaranteed visual pickup demo.",
    )
    add_app_launcher_args(parser)
    args = parser.parse_args()

    simulation_app = launch_simulation_app(args)
    recorder = EpisodeRecorder() if args.record else None
    env = None
    try:
        env = IsaacLabEnvBridge(task=args.task)
        if not args.allow_fallback and not env.is_real_env:
            raise RuntimeError(
                "grasp_cube_demo requires the real Isaac Lab scene by default because the fallback adapter has "
                "no real hand/table/cube USD assets. Re-run with --allow_fallback only if you explicitly want the "
                "synthetic path."
            )
        env.reset()
        cube_center = _current_cube_center(env)
        print(f"[*] Cube center: {cube_center.tolist()}")

        target_dt = 1.0 / max(args.loop_hz, 1.0)
        last_phase = None
        last_action = None
        last_left_pose = None
        last_left_grip = 0.0
        step_idx = 0
        app_running_warned = False
        assist_attached = False
        for phase_name, left_pose, right_pose, left_qpos, right_qpos, left_grip in _iter_actions(cube_center):
            app_running = simulation_app.is_running()
            if not app_running and not app_running_warned:
                print("[Warning] SimulationApp reported not running during scripted demo; "
                      "continuing finite grasp sequence unless stepping fails.")
                app_running_warned = True
            if args.respect_app_running and not app_running:
                print("[Warning] Stopping scripted demo early because --respect_app_running was set.")
                break
            if phase_name != last_phase:
                print(f"[*] Phase: {phase_name}")
                last_phase = phase_name

            action = assemble_teleop_action(left_pose, right_pose, left_qpos, right_qpos)
            last_action = action
            last_left_pose = left_pose.copy()
            last_left_grip = left_grip
            loop_start = time.perf_counter()
            try:
                obs = env.step(action, head_rmat=HEAD_RMAT)
            except Exception as exc:
                print(f"[Error] Demo step failed during phase '{phase_name}' at scripted step {step_idx}: {exc}")
                raise

            if args.assist_cube and phase_name in {"lift", "hold"} and left_grip >= 0.95:
                if not assist_attached:
                    print("[*] Assist cube attachment engaged.")
                    assist_attached = True
                _set_cube_pose(env, _cube_attach_position(left_pose))

            if recorder is not None:
                recorder.append(
                    left_rgb=obs.left_rgb,
                    right_rgb=obs.right_rgb,
                    state=obs.state,
                    action=action,
                    cmd=action,
                )

            elapsed = time.perf_counter() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            step_idx += 1

        if last_action is not None and last_left_pose is not None:
            print("[*] Scripted grasp sequence complete.")
            _hold_final_pose(
                env,
                simulation_app,
                duration_s=args.post_grasp_hold_s,
                loop_hz=args.loop_hz,
                action=last_action,
                left_pose=last_left_pose,
                left_grip=last_left_grip,
                assist_cube=args.assist_cube,
            )
            if args.stay_open:
                _stay_open_until_closed(
                    env,
                    simulation_app,
                    loop_hz=args.loop_hz,
                    action=last_action,
                    left_pose=last_left_pose,
                    left_grip=last_left_grip,
                    assist_cube=args.assist_cube,
                )

        if recorder is not None:
            recorder.save(Path(args.output))
            print(f"[*] Saved scripted grasp episode to: {args.output}")
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
