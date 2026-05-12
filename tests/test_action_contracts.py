import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tv_isaaclab.contracts import (  # noqa: E402
    H1_ACTION_DIM,
    H1_ACTION_SCHEMA,
    H1_LEGACY_ACTION_DIM,
    H1_TASK_ID,
    TELEOP_ACTION_DIM,
    TELEOP_CMD_SCHEMA,
    TELEOP_TASK_ID,
    adapt_h1_action,
    assemble_teleop_action,
    h1_action_to_qpos,
    infer_task_from_episode,
)


class ActionContractTests(unittest.TestCase):
    def test_assemble_teleop_action_keeps_legacy_38d_layout(self):
        left_pose = np.arange(7, dtype=np.float32)
        right_pose = np.arange(7, 14, dtype=np.float32)
        left_qpos = np.arange(12, dtype=np.float32)
        right_qpos = np.arange(12, 24, dtype=np.float32)
        action = assemble_teleop_action(left_pose, right_pose, left_qpos, right_qpos)
        self.assertEqual(action.shape, (TELEOP_ACTION_DIM,))
        np.testing.assert_array_equal(action[:7], left_pose)
        np.testing.assert_array_equal(action[7:14], right_pose)
        np.testing.assert_array_equal(action[14:26], left_qpos)
        np.testing.assert_array_equal(action[26:38], right_qpos)

    def test_h1_adapter_accepts_legacy_28d_and_truncates_to_canonical_26d(self):
        legacy = np.arange(H1_LEGACY_ACTION_DIM, dtype=np.float32)
        adapted = adapt_h1_action(legacy)
        self.assertEqual(adapted.shape, (H1_ACTION_DIM,))
        np.testing.assert_array_equal(adapted, legacy[:H1_ACTION_DIM])

    def test_h1_qpos_conversion_matches_original_hand_spread_layout(self):
        action = np.arange(H1_ACTION_DIM, dtype=np.float32)
        qpos = h1_action_to_qpos(action)
        self.assertEqual(qpos.shape, (51,))
        np.testing.assert_array_equal(qpos[13:20], action[0:7])
        np.testing.assert_array_equal(qpos[32:39], action[13:20])
        self.assertEqual(qpos[20], action[7])
        self.assertEqual(qpos[40], action[20])

    def test_episode_metadata_routes_h1_recordings_to_h1_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_path = Path(tmp_dir) / "episode.hdf5"
            with h5py.File(str(episode_path), "w") as handle:
                handle.create_dataset("qpos_action", data=np.zeros((2, H1_ACTION_DIM), dtype=np.float32))
                handle.attrs["action_schema"] = H1_ACTION_SCHEMA
            self.assertEqual(infer_task_from_episode(episode_path), H1_TASK_ID)

    def test_episode_metadata_routes_teleop_recordings_to_teleop_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_path = Path(tmp_dir) / "episode.hdf5"
            with h5py.File(str(episode_path), "w") as handle:
                handle.create_dataset(
                    "qpos_action", data=np.zeros((2, TELEOP_ACTION_DIM), dtype=np.float32)
                )
                handle.attrs["action_schema"] = TELEOP_CMD_SCHEMA
            self.assertEqual(infer_task_from_episode(episode_path), TELEOP_TASK_ID)

    def test_repo_scripts_reference_schema_aware_routing(self):
        replay_source = (ROOT_DIR / "scripts" / "replay_demo.py").read_text(encoding="utf-8")
        deploy_source = (ROOT_DIR / "scripts" / "deploy_sim.py").read_text(encoding="utf-8")
        collect_source = (ROOT_DIR / "scripts" / "collect_episodes.py").read_text(encoding="utf-8")
        task_source = (ROOT_DIR / "tv_isaaclab" / "tasks" / "television_lab.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("infer_task_from_episode", replay_source)
        self.assertIn("infer_task_from_episode", deploy_source)
        self.assertIn('ROOT_DIR / "teleop"', collect_source)
        self.assertIn("register_television_h1", task_source)

    def test_real_task_module_reuses_shared_action_contracts(self):
        real_task_source = (
            ROOT_DIR / "tv_isaaclab" / "tasks" / "television_lab_real.py"
        ).read_text(encoding="utf-8")
        self.assertIn("assemble_teleop_action", real_task_source)
        self.assertIn("adapt_h1_action", real_task_source)
        self.assertIn("env_cfg_entry_point", real_task_source)

    def test_training_data_loader_discovers_processed_episodes_by_glob(self):
        utils_source = (ROOT_DIR / "act" / "utils.py").read_text(encoding="utf-8")
        self.assertIn('glob("processed_episode_*.hdf5")', utils_source)
        self.assertIn("infer_task_from_schemas", utils_source)
        self.assertIn("def __len__(self):", utils_source)
        self.assertIn("return int(self.cumulative_len[-1])", utils_source)

    def test_headless_full_run_script_exercises_dataset_and_deploy_paths(self):
        source = (ROOT_DIR / "scripts" / "headless_full_run.py").read_text(encoding="utf-8")
        self.assertIn("DummyChunkPolicy", source)
        self.assertIn("EpisodicDataset", source)
        self.assertIn("infer_task_from_episode", source)
        self.assertIn("Player(task=task, show_plot=False)", source)


if __name__ == "__main__":
    unittest.main()
