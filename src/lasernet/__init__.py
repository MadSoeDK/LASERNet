from lasernet._logging import setup_logging

# Auto-configure logging when package is imported
setup_logging()

__all__ = ["setup_logging"]
