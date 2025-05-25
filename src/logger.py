import errno
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import yaml


class ColorFormatter(logging.Formatter):
  COLORS = {
    'VERBOSE': '\033[38;5;52m',
    'INFO': '\033[38;5;91m',
    'WARNING': '\033[38;5;202m',
    'DEBUG': '\033[38;5;18m',
    'ERROR': '\033[38;5;161m',
    'CRITICAL': '\033[38;5;76m',
    'ENDC': '\033[0m',
  }

  def format(self, record):
    return f'{self.COLORS[record.levelname]}{super().format(record)}{self.COLORS["ENDC"]}'


logger = logging.getLogger()  # use the root logger


def init_logger(verbose: bool = False, debug: bool = False) -> None:
  with open('settings.yml') as f:
    logger_config = yaml.safe_load(f)['logger']

  VERBOSE_LEVEL = 15
  logging.addLevelName(VERBOSE_LEVEL, 'VERBOSE')

  def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LEVEL):
      self._log(VERBOSE_LEVEL, message, args, **kwargs)
  logging.Logger.verbose = verbose

  logger.setLevel(logging.INFO)
  if verbose:
    logger.setLevel(VERBOSE_LEVEL)
  if debug:  # debug overrides verbose
    logger.setLevel(logging.DEBUG)

  pattern = '%(asctime)s - %(levelname)s - %(message)s'

  log_path = os.getenv('LOG_FILE')
  if not os.path.exists(os.path.dirname(log_path)):
    try:
      os.makedirs(os.path.dirname(log_path))
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
  file_handler = RotatingFileHandler(log_path, mode='a', maxBytes=logger_config['max_bytes'], backupCount=logger_config['backup_count'])
  file_handler.setFormatter(logging.Formatter(pattern))
  logger.addHandler(file_handler)

  stream_handler = logging.StreamHandler(sys.stdout)
  stream_handler.setFormatter(ColorFormatter(pattern))
  logger.addHandler(stream_handler)
