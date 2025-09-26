import logging
import sys
from pythonjsonlogger import jsonlogger



def init_logging(log_path: str, level=logging.INFO):
    logger = logging.getLogger()

    # Clear existing handlers (important if reloading in notebooks)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # File handler (JSON format)
    file_handler = logging.FileHandler(log_path, mode="a")  # append instead of overwrite
    file_handler.setLevel(level)
    file_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Quiet down noisy libs
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    return logger