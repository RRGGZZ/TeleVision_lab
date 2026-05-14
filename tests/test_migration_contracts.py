import inspect
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class MigrationContractTests(unittest.TestCase):
    def test_env_bridge_imports_os_for_real_isaaclab_path(self):
        import tv_isaaclab.env_bridge as env_bridge

        self.assertTrue(
            hasattr(env_bridge, "os"),
            "IsaacLabEnvBridge uses os.environ on the real Isaac Lab path and must import os.",
        )

    def test_env_bridge_step_accepts_optional_runtime_metadata(self):
        from tv_isaaclab.env_bridge import IsaacLabEnvBridge

        signature = inspect.signature(IsaacLabEnvBridge.step)
        self.assertTrue(
            any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            or "head_rmat" in signature.parameters,
            "IsaacLabEnvBridge.step must accept optional teleoperation metadata such as head_rmat.",
        )

    def test_env_bridge_declares_teleop_capability_contract(self):
        source = (ROOT_DIR / "tv_isaaclab" / "env_bridge.py").read_text(encoding="utf-8")
        self.assertIn(
            "supports_teleop_to_action",
            source,
            "The bridge should declare whether it natively converts teleop signals to env actions.",
        )
        self.assertIn(
            "set_head_rotation",
            source,
            "The bridge should pass teleoperation head-pose metadata into compatible environments.",
        )
        self.assertIn(
            "action_schema",
            source,
            "The bridge should expose the environment action schema to downstream tooling.",
        )
        self.assertIn(
            "task_contract",
            source,
            "The bridge should surface its resolved task contract for replay/training tooling.",
        )
        self.assertIn(
            "array = _as_numpy(img)",
            source,
            "GPU image tensors should be copied to host memory before NumPy/OpenCV use.",
        )
        self.assertIn(
            "state = _as_numpy(self._find_by_keys(obs, self.state_keys))",
            source,
            "GPU state tensors should be copied to host memory before exposing ObsPack.state.",
        )
        self.assertIn(
            "return self._env_target.adapt_action(action)",
            source,
            "Action adaptation should preserve env-native tensor types for the simulator backend.",
        )
        self.assertIn(
            "if self._any_true(terminated) or self._any_true(truncated):",
            source,
            "Termination checks should work for both NumPy arrays and torch tensors.",
        )
        self.assertIn(
            "using direct fallback adapter",
            source,
            "The bridge should fall back to the local adapter env when real Isaac Lab creation fails.",
        )

    def test_deploy_script_has_no_legacy_state_action_shape_constants(self):
        source = (ROOT_DIR / "scripts" / "deploy_sim.py").read_text(encoding="utf-8")
        legacy_fragments = (
            "action_dim = 28",
            "view((1, 26))",
            "view((1, 2, 3, 480, 640))",
        )
        for fragment in legacy_fragments:
            self.assertNotIn(
                fragment,
                source,
                f"Legacy IsaacGym-era shape constant still present: {fragment}",
            )

    def test_training_script_has_no_legacy_state_action_shape_constants(self):
        source = (ROOT_DIR / "act" / "imitate_episodes.py").read_text(encoding="utf-8")
        legacy_fragments = (
            "state_dim = 26",
            "action_dim = 28",
        )
        for fragment in legacy_fragments:
            self.assertNotIn(
                fragment,
                source,
                f"Legacy IsaacGym-era shape constant still present: {fragment}",
            )

    def test_task_does_not_default_to_unrelated_cartpole_scene(self):
        source = (ROOT_DIR / "tv_isaaclab" / "tasks" / "television_lab.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(
            "Isaac-Cartpole-Direct-v0",
            source,
            "television_lab should not silently bind to an unrelated Cartpole task.",
        )
        self.assertIn(
            "if raw_obs is not None:",
            source,
            "Synthetic fallback should preserve teleop-driven state instead of zeroing it out.",
        )
        self.assertIn('self.is_real_env = False', source)
        self.assertIn("self.action_schema =", source)
        self.assertIn("self.state_schema =", source)

    def test_real_task_creates_camera_parent_rig_before_tiled_camera(self):
        source = (ROOT_DIR / "tv_isaaclab" / "tasks" / "television_lab_real.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('prim_path="{ENV_REGEX_NS}/head_cam"', source)
        self.assertIn('"cfg": TeleVisionTeleopEnvCfg()', source)
        self.assertIn('"cfg": TeleVisionH1EnvCfg()', source)
        self.assertIn("self._is_closed = True", source)
        self.assertIn('init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.2))', source)
        self.assertIn('init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.25))', source)
        self.assertIn('self._head_anchor = torch.tensor([-0.6, 0.0, 1.6]', source)
        self.assertIn("_REFERENCE_HAND_QUAT_WXYZ = (0.5, 0.5, -0.5, 0.5)", source)
        self.assertIn('side_offset = 0.5 if side == "left" else -0.5', source)
        self.assertIn("pose[:, 0] = -0.3", source)
        self.assertIn("pose[:, 2] = 1.1", source)
        self.assertIn("viewer: ViewerCfg = ViewerCfg(", source)

    def test_package_prefers_real_task_registration_before_fallback(self):
        source = (ROOT_DIR / "tv_isaaclab" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("television_lab_real", source)
        self.assertIn("fallback_adapter", source)
        self.assertIn("REGISTERED_TASK_BACKEND", source)

    def test_bootstrap_enables_cameras_before_launch(self):
        source = (ROOT_DIR / "tv_isaaclab" / "bootstrap.py").read_text(encoding="utf-8")
        self.assertIn("_configure_camera_mode()", source)
        self.assertIn('os.environ.setdefault("ENABLE_CAMERAS", "1")', source)
        self.assertIn("patch_warp_legacy_api_aliases()", source)
        self.assertIn('setattr(warp_types, "array", wp.array)', source)
        self.assertIn('setattr(wp, "context", wp)', source)
        self.assertIn("_register_real_tasks_after_app()", source)

    def test_headless_smoke_script_defaults_to_headless_app_launcher(self):
        source = (ROOT_DIR / "scripts" / "headless_full_run.py").read_text(encoding="utf-8")
        self.assertIn("parser.set_defaults(headless=True)", source)
        self.assertNotIn('parser.add_argument("--device"', source)
        self.assertIn("--policy_device", source)
        self.assertIn("--memory_mode low --headless", (ROOT_DIR / "README.md").read_text(encoding="utf-8"))
        self.assertIn("--policy_device cuda", (ROOT_DIR / "README.md").read_text(encoding="utf-8"))

    def test_teleop_script_reports_python_313_dex_retargeting_constraint(self):
        source = (ROOT_DIR / "teleop" / "teleop_hand.py").read_text(encoding="utf-8")
        self.assertIn("dex_retargeting is required", source)
        self.assertIn("Python 3.11 or 3.12", source)
        self.assertIn("Python < 3.13", source)
        self.assertIn("dex-retargeting<0.5.0", source)
        self.assertIn("_validate_vuer_certificate_files", source)
        self.assertIn("mkcert -install", source)
        self.assertIn("class MockTeleop", source)
        self.assertIn("--mock_teleop", source)
        self.assertIn("--require_real_env", source)
        self.assertIn("Real Isaac Lab backend", source)
        self.assertIn("_REFERENCE_LEFT_POSE_XYZW", source)
        self.assertIn("_REFERENCE_RIGHT_POSE_XYZW", source)
        self.assertNotIn("\nRetargetingConfig = _load_retargeting_config()", source)

    def test_runtime_diagnostic_reports_warp_state(self):
        source = (ROOT_DIR / "scripts" / "diagnose_isaac_runtime.py").read_text(encoding="utf-8")
        self.assertIn("warp.types", source)
        self.assertIn("patch_warp_legacy_api_aliases", source)
        self.assertIn("ROOT_DIR", source)
        self.assertIn("isaaclab_tasks", source)
        self.assertIn("television_lab_real", source)
        self.assertIn("--skip_runtime", source)
        self.assertIn("REGISTERED_TASK_BACKEND", source)
        self.assertIn("bridge.is_real_env", source)
        self.assertIn("hasattr(warp, 'context')", source)

    def test_requirements_pin_dex_retargeting_to_numpy1_compatible_line(self):
        requirements = (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8")
        install_script = (ROOT_DIR / "install_deps.sh").read_text(encoding="utf-8")
        self.assertIn("dex-retargeting>=0.4.5,<0.5.0", requirements)
        self.assertIn("dex-retargeting>=0.4.5,<0.5.0", install_script)
        self.assertIn("packaging==23.0", requirements)
        self.assertIn("wheel<0.47", requirements)
        self.assertIn("wheel<0.47", install_script)


if __name__ == "__main__":
    unittest.main()
