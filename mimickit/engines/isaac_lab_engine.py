import logging

try:
    from omni.isaac.lab.app import AppLauncher
except ImportError as err:
    AppLauncher = None
    _IMPORT_ERROR = err
else:
    _IMPORT_ERROR = None

from engines import isaac_gym_engine


class IsaacLabEngine(isaac_gym_engine.IsaacGymEngine):
    """
    Drop-in replacement for ``IsaacGymEngine`` that boots the Omniverse
    Isaac Lab application before delegating to the standard Isaac Gym backend.

    Isaac Lab exposes the original Isaac Gym API when launched through
    ``AppLauncher``.  By inheriting from :class:`IsaacGymEngine` we reuse the
    state management, buffer handling, and action application already
    implemented in MimicKit while executing inside the Isaac Lab runtime.
    """

    def __init__(self, config, num_envs, device, visualize, control_mode=None):
        if AppLauncher is None:
            raise RuntimeError(
                "omni.isaac.lab is not available. Install Isaac Lab and make sure "
                "it is on PYTHONPATH."
            ) from _IMPORT_ERROR

        headless = not visualize
        self._app_launcher = AppLauncher(headless=headless)
        self._simulation_app = self._app_launcher.app

        super().__init__(
            config=config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            control_mode=control_mode,
        )

    def close(self):
        """Shut down the Isaac Lab application."""
        if getattr(self, "_app_launcher", None) is not None:
            try:
                self._app_launcher.close()
            except Exception as exc:
                logging.warning("Failed to close Isaac Lab app cleanly: %s", exc)
            finally:
                self._app_launcher = None
                self._simulation_app = None

    def __del__(self):
        self.close()
