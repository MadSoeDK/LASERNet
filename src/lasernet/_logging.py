import logging
import os

def setup_logging() -> None:
    """Configure repo-wide logging from LASERNET_LOG_LEVEL environment variable.

    Reads the LASERNET_LOG_LEVEL environment variable (case-insensitive) and configures
    the root logger with a consistent format. If the variable is not set or contains
    an invalid value, defaults to INFO level.

    Supported levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

    Example:
        # Set via environment
        $ export LASERNET_LOG_LEVEL=DEBUG
        $ python -m lasernet.preprocess

        # Or inline
        $ LASERNET_LOG_LEVEL=DEBUG python -m lasernet.preprocess
    """
    log_level_str = os.getenv("LASERNET_LOG_LEVEL", "INFO").upper()

    # Validate and get log level, fallback to INFO if invalid
    log_level = getattr(logging, log_level_str, None)
    if not isinstance(log_level, int):
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override any existing configuration
    )
