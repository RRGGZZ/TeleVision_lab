"""Diagnose Isaac Sim / Isaac Lab runtime state for TeleVision_lab."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
import traceback
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _try_import(module_name: str) -> tuple[bool, object | None]:
    print(f"\n[check] import {module_name}")
    try:
        module = importlib.import_module(module_name)
    except Exception:
        print("  FAIL")
        traceback.print_exc(limit=5)
        return False, None

    print("  OK")
    print(f"  file: {getattr(module, '__file__', 'built-in/namespace')}")
    return True, module


def _print_package_versions() -> None:
    print("\n=== Installed package versions ===")
    for dist in (
        "isaacsim",
        "isaacsim-core",
        "isaaclab",
        "isaaclab-tasks",
        "warp",
        "warp-lang",
        "torch",
        "numpy",
        "packaging",
    ):
        print(f"{dist}: {_dist_version(dist)}")


def _check_warp_alias() -> None:
    warp_ok, _ = _try_import("warp")
    if not warp_ok:
        return

    import warp as wp

    print(f"  warp.__version__: {getattr(wp, '__version__', 'unknown')}")
    print(f"  hasattr(warp, 'array'): {hasattr(wp, 'array')}")
    print(f"  hasattr(warp.types, 'array'): {hasattr(getattr(wp, 'types', None), 'array')}")
    if not hasattr(getattr(wp, "types", None), "array"):
        print(
            "  PROBLEM: Isaac Sim is importing a Warp build without warp.types.array. "
            "This usually means the active environment has an incompatible or shadowing Warp package."
        )
        try:
            from tv_isaaclab.bootstrap import patch_warp_legacy_array_alias

            patched = patch_warp_legacy_array_alias()
            print(f"  compatibility shim applied: {patched}")
            print(f"  after shim hasattr(warp.types, 'array'): {hasattr(getattr(wp, 'types', None), 'array')}")
        except Exception:
            print("  compatibility shim failed")
            traceback.print_exc(limit=3)


def _prelaunch_checks() -> None:
    print("\n=== Prelaunch Checks ===")
    _check_warp_alias()
    _try_import("isaaclab.app")
    print("\n[note] `pxr`, `isaacsim.core`, and many Isaac extensions may stay unavailable")
    print("       until SimulationApp / Kit has been launched. Prelaunch import failure there")
    print("       is not, by itself, proof that the runtime is broken.")


def _runtime_checks(task: str, headless: bool, memory_mode: str) -> int:
    print("\n=== Runtime Checks ===")
    try:
        from tv_isaaclab import IsaacLabEnvBridge, add_app_launcher_args, launch_simulation_app
        import tv_isaaclab
    except Exception:
        traceback.print_exc(limit=5)
        return 1

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default=task)
    add_app_launcher_args(parser)
    arg_list = ["--task", task, "--memory_mode", memory_mode]
    if headless:
        arg_list.append("--headless")
    args = parser.parse_args(arg_list)

    simulation_app = None
    env = None
    try:
        simulation_app = launch_simulation_app(args)
        print(f"  REGISTERED_TASK_BACKEND: {getattr(tv_isaaclab, 'REGISTERED_TASK_BACKEND', None)}")

        _try_import("pxr")
        _try_import("isaacsim.core.utils.warp.rotations")
        _try_import("isaaclab_tasks")
        _try_import("tv_isaaclab.tasks.television_lab_real")

        env = IsaacLabEnvBridge(task=task)
        print(f"  bridge.task: {env.task}")
        print(f"  bridge.is_real_env: {env.is_real_env}")
        print(f"  bridge.action_schema: {env.action_schema}")
        print(f"  bridge.state_schema: {env.state_schema}")

        obs = env.reset()
        print(f"  reset left_rgb shape: {obs.left_rgb.shape}")
        print(f"  reset right_rgb shape: {obs.right_rgb.shape}")
        print(f"  reset state shape: {obs.state.shape}")
        return 0
    except Exception:
        traceback.print_exc(limit=8)
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Isaac Sim / Isaac Lab runtime for TeleVision_lab")
    parser.add_argument("--task", default="television_lab")
    parser.add_argument("--skip_runtime", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--memory_mode", default="low", choices=["auto", "low", "medium", "high"])
    args = parser.parse_args()

    print("=== TeleVision_lab Isaac Runtime Diagnostic ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    _print_package_versions()
    _prelaunch_checks()

    if args.skip_runtime:
        print("\n=== Next step ===")
        print("Run again without `--skip_runtime` to launch SimulationApp and verify the real runtime path.")
        return 0

    rc = _runtime_checks(task=args.task, headless=args.headless, memory_mode=args.memory_mode)
    print("\n=== Next step ===")
    if rc == 0:
        print("If `bridge.is_real_env: True`, the real Isaac scene path is active.")
    else:
        print("If runtime checks failed, the active Isaac Sim / Isaac Lab runtime still needs repair.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
