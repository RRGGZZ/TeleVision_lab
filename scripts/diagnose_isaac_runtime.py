"""Diagnose Isaac Sim / Isaac Lab import state for TeleVision_lab."""

from __future__ import annotations

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


def _try_import(module_name: str) -> bool:
    print(f"\n[check] import {module_name}")
    try:
        module = importlib.import_module(module_name)
    except Exception:
        print("  FAIL")
        traceback.print_exc(limit=5)
        return False

    print("  OK")
    print(f"  file: {getattr(module, '__file__', 'built-in/namespace')}")
    return True


def main() -> int:
    print("=== TeleVision_lab Isaac Runtime Diagnostic ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")

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

    warp_ok = _try_import("warp")
    if warp_ok:
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

    _try_import("isaacsim.core.utils.warp.rotations")
    _try_import("isaaclab.app")
    _try_import("isaaclab_tasks")
    _try_import("tv_isaaclab.tasks.television_lab_real")

    print("\n=== Next step ===")
    print(
        "If any Isaac imports fail, repair the active conda/pip environment before expecting "
        "real USD assets to render. The fallback adapter can still run data-path smoke tests, "
        "but it is not a visual Isaac scene."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
