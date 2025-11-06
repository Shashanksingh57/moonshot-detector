"""
Logging configuration for moonshot-detector.
Sets up structured logging with console and file handlers.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(
    module_name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    config_path: str = "config.yaml"
) -> logging.Logger:
    """
    Set up logging for a module.

    Args:
        module_name: Name of the module (e.g., 'data_collection')
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional specific log file path
        config_path: Path to config.yaml

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Load config if available
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            log_level = config.get('logging', {}).get('level', log_level)
            log_files = config.get('logging', {}).get('files', {})
            if not log_file and module_name in log_files:
                log_file = log_files[module_name]
    except FileNotFoundError:
        pass

    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get or create a logger for a module.

    Args:
        module_name: Name of the module

    Returns:
        Logger instance
    """
    logger = logging.getLogger(module_name)

    # If logger has no handlers, set it up
    if not logger.handlers:
        logger = setup_logging(module_name)

    return logger
