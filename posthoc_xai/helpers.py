import os
import logging
import colorlog


def initialize_log():
    pid = os.getpid()
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter())

    logging.basicConfig(handlers=[handler])
    log = logging.getLogger(f'posthoc_xai-{pid}')
    log_level = os.environ.get('POSTHOC_XAI_LOG', 'DEBUG')
    log.setLevel(log_level)
    return log


log = initialize_log()
