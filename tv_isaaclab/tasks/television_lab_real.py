from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_matrix

from tv_isaaclab.contracts import (
    H1_ACTION_DIM,
    H1_ACTION_SCHEMA,
    H1_STATE_SCHEMA,
    H1_TASK_ID,
    TELEOP_ACTION_DIM,
    TELEOP_CMD_SCHEMA,
    TELEOP_STATE_SCHEMA,
    TELEOP_TASK_ID,
    adapt_h1_action,
    assemble_teleop_action,
)


_ROOT_DIR = Path(__file__).resolve().parents[2]
_ASSET_DIR = _ROOT_DIR / "assets"
_LEFT_HAND_URDF = (_ASSET_DIR / "inspire_hand" / "inspire_hand_left.urdf").as_posix()
_RIGHT_HAND_URDF = (_ASSET_DIR / "inspire_hand" / "inspire_hand_right.urdf").as_posix()
_H1_URDF = (_ASSET_DIR / "h1_inspire" / "urdf" / "h1_inspire.urdf").as_posix()


def _make_urdf_cfg(
    asset_path: str,
    *,
    pd_stiffness: float,
    pd_damping: float,
    convert_mimic_joints_to_normal_joints: bool,
    root_link_name: str | None = None,
):
    """Build a URDF spawn config while tolerating minor Isaac Lab API drift."""
    kwargs = {
        "asset_path": asset_path,
        "fix_base": True,
        "merge_fixed_joints": True,
        "convert_mimic_joints_to_normal_joints": convert_mimic_joints_to_normal_joints,
        "self_collision": False,
        "joint_drive": sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=pd_stiffness,
                damping=pd_damping,
            ),
        ),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
        ),
        "articulation_props": sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    }
    if root_link_name:
        kwargs["root_link_name"] = root_link_name
    try:
        return sim_utils.UrdfFileCfg(**kwargs)
    except TypeError as exc:
        if "root_link_name" in str(exc):
            kwargs.pop("root_link_name", None)
            return sim_utils.UrdfFileCfg(**kwargs)
        raise


def _camera_offset_cfg():
    kwargs = {
        "pos": (0.0, 0.0, 0.0),
        "rot": (1.0, 0.0, 0.0, 0.0),
        "convention": "world",
    }
    try:
        return TiledCameraCfg.OffsetCfg(**kwargs)
    except TypeError:
        kwargs.pop("convention", None)
        return TiledCameraCfg.OffsetCfg(**kwargs)


def _teleop_hand_cfg(prim_path: str, asset_path: str, root_link_name: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=_make_urdf_cfg(
            asset_path=asset_path,
            pd_stiffness=200.0,
            pd_damping=10.0,
            convert_mimic_joints_to_normal_joints=True,
            root_link_name=root_link_name,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            )
        },
    )


def _h1_cfg(prim_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=_make_urdf_cfg(
            asset_path=_H1_URDF,
            pd_stiffness=200.0,
            pd_damping=20.0,
            convert_mimic_joints_to_normal_joints=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.8, 0.0, 1.1)),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            )
        },
    )


def _stereo_camera_cfg(width: int, height: int) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/HeadStereo/head_cam",
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=_camera_offset_cfg(),
    )


def _table_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5), metallic=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.2)),
    )


def _cube_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.5), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.25)),
    )


@configclass
class TeleVisionTeleopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    left_hand: ArticulationCfg = _teleop_hand_cfg(
        prim_path="{ENV_REGEX_NS}/LeftHand",
        asset_path=_LEFT_HAND_URDF,
        root_link_name="L_hand_base_link",
    )
    right_hand: ArticulationCfg = _teleop_hand_cfg(
        prim_path="{ENV_REGEX_NS}/RightHand",
        asset_path=_RIGHT_HAND_URDF,
        root_link_name="R_hand_base_link",
    )
    table = _table_cfg()
    cube = _cube_cfg()
    stereo_camera = _stereo_camera_cfg(width=1280, height=720)
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class TeleVisionH1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = _h1_cfg(prim_path="{ENV_REGEX_NS}/Robot")
    table = _table_cfg()
    cube = _cube_cfg()
    stereo_camera = _stereo_camera_cfg(width=1280, height=720)
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class TeleVisionTeleopEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 60.0
    action_space = TELEOP_ACTION_DIM
    observation_space = {
        "policy": TELEOP_ACTION_DIM,
        "observation": {
            "state": TELEOP_ACTION_DIM,
            "image": {"left": [720, 1280, 3], "right": [720, 1280, 3]},
        },
    }
    state_space = TELEOP_ACTION_DIM
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1 / 60, render_interval=decimation)
    scene: InteractiveSceneCfg = TeleVisionTeleopSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
        filter_collisions=True,
    )
    num_rerenders_on_reset = 2


@configclass
class TeleVisionH1EnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 60.0
    action_space = H1_ACTION_DIM
    observation_space = {
        "policy": H1_ACTION_DIM,
        "observation": {
            "state": H1_ACTION_DIM,
            "image": {"left": [720, 1280, 3], "right": [720, 1280, 3]},
        },
    }
    state_space = H1_ACTION_DIM
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1 / 60, render_interval=decimation)
    scene: InteractiveSceneCfg = TeleVisionH1SceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
        filter_collisions=True,
    )
    num_rerenders_on_reset = 2


class _TeleVisionDirectEnvBase(DirectRLEnv):
    state_dim: int
    action_dim: int
    is_real_env: bool = True
    supports_teleop_to_action: bool = False
    action_schema: str
    state_schema: str

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        self._latest_action = None
        self._head_rmat = None
        self._eye_offset = None
        self._head_anchor = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._latest_action = torch.zeros((self.num_envs, self.action_dim), dtype=torch.float32, device=self.device)
        self._head_rmat = torch.eye(3, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1, 1)
        self._eye_offset = torch.tensor([0.0, 0.033, 0.0], dtype=torch.float32, device=self.device)
        self._head_anchor = torch.tensor([-0.6, 0.0, 1.6], dtype=torch.float32, device=self.device)
        self._zero_root_velocity = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

    def adapt_action(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[-1]}.")
        return action

    def set_head_rotation(self, head_rmat) -> None:
        matrix = torch.as_tensor(head_rmat, dtype=torch.float32, device=self.device)
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0).repeat(self.num_envs, 1, 1)
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Expected head rotation with shape (*, 3, 3), got {tuple(matrix.shape)}.")
        if matrix.shape[0] == 1 and self.num_envs > 1:
            matrix = matrix.repeat(self.num_envs, 1, 1)
        self._head_rmat = matrix

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._latest_action = self.adapt_action(actions).clone()

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _render_stereo_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_pos, left_quat = self._camera_pose_from_head(eye_sign=1.0)
        self.stereo_camera.set_world_poses(left_pos, left_quat, convention="world")
        self.sim.render()
        self.stereo_camera.update(0.0, force_recompute=True)
        left_rgb = self._camera_rgb().clone()

        right_pos, right_quat = self._camera_pose_from_head(eye_sign=-1.0)
        self.stereo_camera.set_world_poses(right_pos, right_quat, convention="world")
        self.sim.render()
        self.stereo_camera.update(0.0, force_recompute=True)
        right_rgb = self._camera_rgb().clone()
        return left_rgb, right_rgb

    def _camera_pose_from_head(self, eye_sign: float) -> tuple[torch.Tensor, torch.Tensor]:
        rotated_offset = torch.matmul(self._head_rmat, (self._eye_offset * eye_sign).view(1, 3, 1)).squeeze(-1)
        position = self.scene.env_origins + self._head_anchor.unsqueeze(0) + rotated_offset
        orientation = quat_from_matrix(self._head_rmat)
        return position, orientation

    def _set_debug_vis_impl(self, debug_vis: bool):
        return None

    def _camera_rgb(self) -> torch.Tensor:
        outputs = getattr(self.stereo_camera.data, "output", {})
        for key in ("rgb", "rgba"):
            if key in outputs:
                image = outputs[key]
                return image[..., :3] if image.shape[-1] == 4 else image
        raise KeyError("TeleVision stereo camera did not expose an rgb/rgba output.")


class TeleVisionTeleopDirectEnv(_TeleVisionDirectEnvBase):
    cfg: TeleVisionTeleopEnvCfg
    state_dim = TELEOP_ACTION_DIM
    action_dim = TELEOP_ACTION_DIM
    supports_teleop_to_action = True
    action_schema = TELEOP_CMD_SCHEMA
    state_schema = TELEOP_STATE_SCHEMA

    def _setup_scene(self):
        self.left_hand: Articulation = self.scene["left_hand"]
        self.right_hand: Articulation = self.scene["right_hand"]
        self.cube: RigidObject = self.scene["cube"]
        self.stereo_camera: TiledCamera = self.scene["stereo_camera"]

    def teleop_to_action(self, left_pose, right_pose, left_qpos, right_qpos):
        return assemble_teleop_action(left_pose, right_pose, left_qpos, right_qpos)

    def _apply_action(self) -> None:
        action = self._latest_action
        left_pose = self._xyzw_pose_to_wxyz(action[:, 0:7].clone())
        right_pose = self._xyzw_pose_to_wxyz(action[:, 7:14].clone())
        left_pose[:, :3] += self.scene.env_origins
        right_pose[:, :3] += self.scene.env_origins
        self.left_hand.write_root_pose_to_sim(left_pose)
        self.right_hand.write_root_pose_to_sim(right_pose)
        self.left_hand.write_root_velocity_to_sim(self._zero_root_velocity)
        self.right_hand.write_root_velocity_to_sim(self._zero_root_velocity)

        left_joint_pos = self._fit_joint_command(action[:, 14:26], self.left_hand.num_joints)
        right_joint_pos = self._fit_joint_command(action[:, 26:38], self.right_hand.num_joints)
        self.left_hand.write_joint_state_to_sim(left_joint_pos, torch.zeros_like(left_joint_pos))
        self.right_hand.write_joint_state_to_sim(right_joint_pos, torch.zeros_like(right_joint_pos))

    def _get_observations(self) -> dict:
        left_rgb, right_rgb = self._render_stereo_pair()
        state = self._latest_action.clone()
        return {
            "policy": state,
            "observation": {
                "state": state,
                "image": {
                    "left": left_rgb,
                    "right": right_rgb,
                },
            },
        }

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)

        left_pose = self._neutral_hand_pose(side="left", env_ids=env_ids)
        right_pose = self._neutral_hand_pose(side="right", env_ids=env_ids)
        self.left_hand.write_root_pose_to_sim(left_pose, env_ids=env_ids)
        self.right_hand.write_root_pose_to_sim(right_pose, env_ids=env_ids)
        self.left_hand.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)
        self.right_hand.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        left_joint_pos = torch.zeros((len(env_ids), self.left_hand.num_joints), device=self.device)
        right_joint_pos = torch.zeros((len(env_ids), self.right_hand.num_joints), device=self.device)
        self.left_hand.write_joint_state_to_sim(left_joint_pos, torch.zeros_like(left_joint_pos), env_ids=env_ids)
        self.right_hand.write_joint_state_to_sim(right_joint_pos, torch.zeros_like(right_joint_pos), env_ids=env_ids)

        cube_pose = self.cube.data.default_root_state[env_ids, :7].clone()
        cube_pose[:, :3] += self.scene.env_origins[env_ids]
        self.cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
        self.cube.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)
        self._latest_action[env_ids] = 0.0

    def _neutral_hand_pose(self, side: str, env_ids: Sequence[int]) -> torch.Tensor:
        pose = torch.zeros((len(env_ids), 7), dtype=torch.float32, device=self.device)
        pose[:, 0] = -0.35
        pose[:, 1] = 0.18 if side == "left" else -0.18
        pose[:, 2] = 1.35
        pose[:, 3] = 1.0
        pose[:, :3] += self.scene.env_origins[env_ids]
        return pose

    @staticmethod
    def _xyzw_pose_to_wxyz(pose: torch.Tensor) -> torch.Tensor:
        reordered = pose.clone()
        reordered[:, 3] = pose[:, 6]
        reordered[:, 4] = pose[:, 3]
        reordered[:, 5] = pose[:, 4]
        reordered[:, 6] = pose[:, 5]
        return reordered

    @staticmethod
    def _fit_joint_command(command: torch.Tensor, joint_dim: int) -> torch.Tensor:
        if command.shape[-1] == joint_dim:
            return command
        fitted = torch.zeros((command.shape[0], joint_dim), dtype=command.dtype, device=command.device)
        copy_n = min(command.shape[-1], joint_dim)
        fitted[:, :copy_n] = command[:, :copy_n]
        return fitted


class TeleVisionH1DirectEnv(_TeleVisionDirectEnvBase):
    cfg: TeleVisionH1EnvCfg
    state_dim = H1_ACTION_DIM
    action_dim = H1_ACTION_DIM
    supports_teleop_to_action = False
    action_schema = H1_ACTION_SCHEMA
    state_schema = H1_STATE_SCHEMA

    def _setup_scene(self):
        self.robot: Articulation = self.scene["robot"]
        self.cube: RigidObject = self.scene["cube"]
        self.stereo_camera: TiledCamera = self.scene["stereo_camera"]

    def adapt_action(self, action):
        action = adapt_h1_action(action)
        return super().adapt_action(action)

    def _apply_action(self) -> None:
        joint_pos = self._compact_h1_action_to_qpos(self._latest_action, self.robot.num_joints)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    def _get_observations(self) -> dict:
        left_rgb, right_rgb = self._render_stereo_pair()
        state = self._latest_action.clone()
        return {
            "policy": state,
            "observation": {
                "state": state,
                "image": {
                    "left": left_rgb,
                    "right": right_rgb,
                },
            },
        }

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int64)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        cube_pose = self.cube.data.default_root_state[env_ids, :7].clone()
        cube_pose[:, :3] += self.scene.env_origins[env_ids]
        self.cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
        self.cube.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)
        self._latest_action[env_ids] = 0.0

    @staticmethod
    def _compact_h1_action_to_qpos(action: torch.Tensor, joint_dim: int) -> torch.Tensor:
        """Mirror tv_isaaclab.contracts.h1_action_to_qpos without leaving torch."""
        qpos = torch.zeros((action.shape[0], joint_dim), dtype=action.dtype, device=action.device)
        copy_n = min(joint_dim, 51)
        compact = action[:, :H1_ACTION_DIM]
        if copy_n >= 20:
            qpos[:, 13:20] = compact[:, 0:7]
        if copy_n >= 22:
            qpos[:, 20:22] = compact[:, 7:8]
        if copy_n >= 24:
            qpos[:, 22:24] = compact[:, 8:9]
        if copy_n >= 26:
            qpos[:, 24:26] = compact[:, 9:10]
        if copy_n >= 28:
            qpos[:, 26:28] = compact[:, 10:11]
        if copy_n >= 29:
            qpos[:, 28] = compact[:, 11]
        if copy_n >= 32:
            qpos[:, 29:32] = compact[:, 12:13] * torch.tensor([1.0, 1.6, 2.4], device=action.device)
        if copy_n >= 39:
            qpos[:, 32:39] = compact[:, 13:20]
        if copy_n >= 41:
            qpos[:, 39:41] = compact[:, 20:21]
        if copy_n >= 43:
            qpos[:, 41:43] = compact[:, 21:22]
        if copy_n >= 45:
            qpos[:, 43:45] = compact[:, 22:23]
        if copy_n >= 47:
            qpos[:, 45:47] = compact[:, 23:24]
        if copy_n >= 48:
            qpos[:, 47] = compact[:, 24]
        if copy_n >= 51:
            qpos[:, 48:51] = compact[:, 25:26] * torch.tensor([1.0, 1.6, 2.4], device=action.device)
        return qpos


def register_television_lab_real():
    gym.register(
        id=TELEOP_TASK_ID,
        entry_point="tv_isaaclab.tasks.television_lab_real:TeleVisionTeleopDirectEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tv_isaaclab.tasks.television_lab_real:TeleVisionTeleopEnvCfg",
        },
    )


def register_television_h1_real():
    gym.register(
        id=H1_TASK_ID,
        entry_point="tv_isaaclab.tasks.television_lab_real:TeleVisionH1DirectEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tv_isaaclab.tasks.television_lab_real:TeleVisionH1EnvCfg",
        },
    )
