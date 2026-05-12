"""Isaac Lab integration utilities for TeleVision."""

from .bootstrap import add_app_launcher_args, launch_simulation_app
from .env_bridge import IsaacLabEnvBridge
from .recording import EpisodeRecorder

REGISTERED_TASK_BACKEND = None


def _register_tasks() -> None:
    global REGISTERED_TASK_BACKEND

    registration_attempts = (
        (
            "isaaclab_direct",
            ".tasks.television_lab_real",
            ("register_television_lab_real", "register_television_h1_real"),
        ),
        (
            "fallback_adapter",
            ".tasks.television_lab",
            ("register_television_lab", "register_television_h1"),
        ),
    )

    errors: list[tuple[str, Exception]] = []
    for backend_name, module_name, register_names in registration_attempts:
        try:
            module = __import__(f"{__name__}{module_name}", fromlist=list(register_names))
            for register_name in register_names:
                getattr(module, register_name)()
            REGISTERED_TASK_BACKEND = backend_name
            return
        except Exception as exc:
            errors.append((backend_name, exc))

    if errors:
        details = "; ".join(f"{backend_name}: {exc}" for backend_name, exc in errors)
        print(f"[Warning] Failed to register TeleVision tasks: {details}")


_register_tasks()

__all__ = [
    "add_app_launcher_args",
    "launch_simulation_app",
    "IsaacLabEnvBridge",
    "EpisodeRecorder",
    "REGISTERED_TASK_BACKEND",
]
