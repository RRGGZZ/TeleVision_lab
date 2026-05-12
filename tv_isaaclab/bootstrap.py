import importlib
import os


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


def launch_simulation_app(args):
    _configure_memory_mode(args)
    _configure_camera_mode()

    app_launcher_cls = _import_app_launcher()
    app_launcher = app_launcher_cls(args)
    return app_launcher.app
