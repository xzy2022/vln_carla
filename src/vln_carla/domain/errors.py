class EnvError(Exception):
    """Base error for environment failures."""


class EnvConnectionError(EnvError):
    """Raised when CARLA connection fails or times out."""


class EnvStepError(EnvError):
    """Raised when a step/tick or sensor sync fails."""
