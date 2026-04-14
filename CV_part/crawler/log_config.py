import logging
import logging.config
import sys
from logging import handlers
from pathlib import Path


def work_config(log_queue):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "queue": {
                    "class": "logging.handlers.QueueHandler",
                    "queue": log_queue,
                }
            },
            "root": {
                "level": "DEBUG",
                "handlers": ["queue"],
            },
        }
    )


def listener_config(log_queue, log_file_path):
    log_file_path = Path(log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = handlers.RotatingFileHandler(
        filename=log_file_path,
        mode="a",
        encoding="utf-8",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - PID:%(process)d - %(levelname)s - %(name)s[line:%(lineno)d] - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    listener = handlers.QueueListener(log_queue, console_handler, file_handler, respect_handler_level=True)
    listener.start()
    return listener
