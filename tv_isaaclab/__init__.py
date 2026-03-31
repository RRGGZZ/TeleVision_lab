"""Isaac Lab integration utilities for TeleVision."""

from .bootstrap import add_app_launcher_args, launch_simulation_app
from .env_bridge import IsaacLabEnvBridge
from .recording import EpisodeRecorder

# Register custom tasks
try:
    from .tasks.television_lab import register_television_lab
    register_television_lab()
except Exception as e:
    print(f"[Warning] Failed to register television_lab task: {e}")

__all__ = [
    "add_app_launcher_args",
    "launch_simulation_app",
    "IsaacLabEnvBridge",
    "EpisodeRecorder",
]
