"""Batch episode collection for the teleoperation scene."""

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from teleop.teleop_hand import ActionMapper, VuerTeleop
from tv_isaaclab import (
    EpisodeRecorder,
    IsaacLabEnvBridge,
    add_app_launcher_args,
    launch_simulation_app,
)
from tv_isaaclab.contracts import TELEOP_CMD_SCHEMA, TELEOP_STATE_SCHEMA, TELEOP_TASK_ID


def _fit_action_dim(action: np.ndarray, expected_dim: int) -> np.ndarray:
    if action.shape[0] == expected_dim:
        return action
    resized = np.zeros(expected_dim, dtype=np.float32)
    copy_n = min(expected_dim, action.shape[0])
    resized[:copy_n] = action[:copy_n]
    return resized


def collect_episodes(
    num_episodes: int,
    task: str,
    output_dir: Path,
    retarget_config: str = "inspire_hand.yml",
    action_mapping: str = "teleop_action_mapping_isaaclab.yml",
    max_steps_per_episode: int = 0,
    ngrok: bool = False,
    simulation_app=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teleoperator = VuerTeleop((ROOT_DIR / "teleop" / retarget_config).as_posix(), ngrok=ngrok)
    mapper = ActionMapper((ROOT_DIR / "teleop" / action_mapping).as_posix())
    env = IsaacLabEnvBridge(task=task)
    env.reset()

    print(f"\n[*] Starting batch collection of {num_episodes} episodes")
    print(f"    Output directory: {output_dir}")
    print(f"    Task: {task}")
    print(f"    Teleop mapping dim: {mapper.action_dim}")
    print(f"    Environment dim: {env.action_dim}")

    successful_episodes = 0
    failed_episodes = 0

    for episode_idx in range(num_episodes):
        if simulation_app and not simulation_app.is_running():
            print("[!] Simulation app closed, stopping collection")
            break

        print(f"\n[Episode {episode_idx + 1}/{num_episodes}]")
        recorder = EpisodeRecorder(
            action_schema=TELEOP_CMD_SCHEMA,
            cmd_schema=TELEOP_CMD_SCHEMA,
            state_schema=TELEOP_STATE_SCHEMA,
        )
        episode_file = output_dir / f"episode_{episode_idx:04d}.hdf5"
        env.reset()
        step_count = 0

        try:
            while simulation_app.is_running() if simulation_app else True:
                head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
                raw_action = mapper.assemble(left_pose, right_pose, left_qpos, right_qpos)
                if getattr(env, "supports_teleop_to_action", False):
                    action = env.teleop_to_action(left_pose, right_pose, left_qpos, right_qpos)
                else:
                    action = _fit_action_dim(raw_action, env.action_dim)

                obs = env.step(action, head_rmat=head_rmat)
                recorder.append(
                    left_rgb=obs.left_rgb,
                    right_rgb=obs.right_rgb,
                    state=obs.state,
                    action=action,
                    cmd=raw_action,
                )

                step_count += 1
                if max_steps_per_episode > 0 and step_count >= max_steps_per_episode:
                    print(f"  Reached max steps ({max_steps_per_episode})")
                    break

        except KeyboardInterrupt:
            if step_count > 0:
                recorder.save(episode_file)
                successful_episodes += 1
                print(f"  Saved episode with {step_count} steps")
            break
        except Exception as exc:
            print(f"  Error during episode: {exc}")
            failed_episodes += 1
            continue

        if step_count > 0:
            recorder.save(episode_file)
            successful_episodes += 1
            print(f"  Saved episode with {step_count} steps -> {episode_file}")
        else:
            failed_episodes += 1
            print("  Episode had no steps")

    env.close()
    print("\n[*] Collection complete!")
    print(f"    Successful: {successful_episodes}/{num_episodes}")
    print(f"    Failed: {failed_episodes}/{num_episodes}")
    print(f"    Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch collect teleoperation episodes")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to collect")
    parser.add_argument("--task", type=str, default=TELEOP_TASK_ID, help="Isaac Lab task name")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/recordings/isaaclab",
        help="Output directory for episodes",
    )
    parser.add_argument("--retarget_config", type=str, default="inspire_hand.yml")
    parser.add_argument("--action_mapping", type=str, default="teleop_action_mapping_isaaclab.yml")
    parser.add_argument("--max_steps", type=int, default=0, help="Max steps per episode (0=unlimited)")
    parser.add_argument("--ngrok", action="store_true", help="Enable ngrok mode")
    add_app_launcher_args(parser)
    args = parser.parse_args()

    simulation_app = None
    try:
        simulation_app = launch_simulation_app(args)
    except Exception as exc:
        print(f"[!] Could not launch simulation app: {exc}")
        print("    Will attempt to use environment directly")

    collect_episodes(
        num_episodes=args.num_episodes,
        task=args.task,
        output_dir=args.output_dir,
        retarget_config=args.retarget_config,
        action_mapping=args.action_mapping,
        max_steps_per_episode=args.max_steps,
        ngrok=args.ngrok,
        simulation_app=simulation_app,
    )
