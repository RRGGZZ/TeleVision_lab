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
    # Add custom args for memory optimization
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

    # Set environment variables for memory optimization
    if args.memory_mode == "low" or args.headless:
        # Headless mode - minimal GPU memory
        os.environ.setdefault("OMNI_KIT_RENDERER_MEMORY", "2048")
        os.environ.setdefault("CARB_GPU_MEMORY", "2048")
        # Force headless if not explicitly set
        args.headless = True
    elif args.memory_mode == "medium":
        os.environ.setdefault("OMNI_KIT_RENDERER_MEMORY", "4096")
        os.environ.setdefault("CARB_GPU_MEMORY", "4096")
    # "high" uses default (all available)


def launch_simulation_app(args):
    # Configure memory mode before launching
    _configure_memory_mode(args)

    app_launcher_cls = _import_app_launcher()
    app_launcher = app_launcher_cls(args)
    return app_launcher.app
