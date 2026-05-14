import importlib
import os
import sys


def _import_app_launcher():
    """Support Isaac Lab API names across recent releases."""
    try:
        module = importlib.import_module("isaaclab.app")
        return module.AppLauncher
    except ModuleNotFoundError:
        module = importlib.import_module("omni.isaac.lab.app")
        return module.AppLauncher


def add_app_launcher_args(parser):
    app_launcher_cls = _import_app_launcher()
    app_launcher_cls.add_app_launcher_args(parser)
    parser.add_argument(
        "--memory_mode",
        type=str,
        default="auto",
        choices=["auto", "low", "medium", "high"],
        help="Memory allocation mode: auto (detect), low (headless), medium, high",
    )
    return parser


def _configure_memory_mode(args):
    """Configure Isaac Sim for different memory modes."""
    if not hasattr(args, "memory_mode"):
        return

    if args.memory_mode == "low":
        os.environ.setdefault("OMNI_KIT_RENDERER_MEMORY", "2048")
        os.environ.setdefault("CARB_GPU_MEMORY", "2048")
    elif args.memory_mode == "medium":
        os.environ.setdefault("OMNI_KIT_RENDERER_MEMORY", "4096")
        os.environ.setdefault("CARB_GPU_MEMORY", "4096")


def _configure_camera_mode():
    """Enable camera pipelines before AppLauncher constructs the simulator."""
    os.environ.setdefault("ENABLE_CAMERAS", "1")


def patch_warp_legacy_array_alias() -> bool:
    """Keep Isaac Sim 5.1 extensions compatible with newer NVIDIA Warp builds."""
    try:
        import warp as wp
    except Exception:
        return False

    warp_types = getattr(wp, "types", None)
    if warp_types is None or hasattr(warp_types, "array") or not hasattr(wp, "array"):
        return False

    setattr(warp_types, "array", wp.array)
    return True


def _register_real_tasks_after_app() -> None:
    """Retry real task registration after Kit/Isaac Sim has populated extension paths."""
    try:
        module = importlib.import_module("tv_isaaclab.tasks.television_lab_real")
        module.register_television_lab_real()
        module.register_television_h1_real()
        package = sys.modules.get("tv_isaaclab")
        if package is not None:
            setattr(package, "REGISTERED_TASK_BACKEND", "isaaclab_direct")
    except Exception as exc:
        print(f"[Warning] Real TeleVision Isaac Lab task registration unavailable after app launch: {exc}")


def launch_simulation_app(args):
    _configure_memory_mode(args)
    _configure_camera_mode()
    if patch_warp_legacy_array_alias():
        print("[*] Applied Warp compatibility shim: warp.types.array -> warp.array")

    app_launcher_cls = _import_app_launcher()
    app_launcher = app_launcher_cls(args)
    _register_real_tasks_after_app()
    return app_launcher.app
