import errno
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any, cast

from . import setting_dict

VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, 'VERBOSE')


class VerboseLogger(logging.Logger):
  def verbose(self, msg: object, *args: object, **kwargs: Any) -> None:
    if self.isEnabledFor(VERBOSE_LEVEL):
      self._log(VERBOSE_LEVEL, msg, args, **kwargs)


logging.setLoggerClass(VerboseLogger)


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


logger = cast(VerboseLogger, logging.getLogger('stylo_flora'))
for handler in logger.handlers:
  logger.removeHandler(handler)
logger.setLevel(logging.CRITICAL)


def init_logger(
    path: os.PathLike | None = None,
    *,
    verbose: bool = False,
    debug: bool = False
) -> None:
  logger.setLevel(logging.INFO)
  if verbose:
    logger.setLevel(VERBOSE_LEVEL)
  if debug:  # debug overrides verbose
    logger.setLevel(logging.DEBUG)

  pattern = '%(asctime)s - %(levelname)s - %(message)s'

  stream_handler = logging.StreamHandler(sys.stdout)
  stream_handler.setFormatter(ColorFormatter(pattern))
  logger.addHandler(stream_handler)

  if not path:
    logger.info('Log path not set, logging to stdout only.')
    return
  if not os.path.exists(os.path.dirname(path)):
    try:
      os.makedirs(os.path.dirname(path))
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise
  file_handler = RotatingFileHandler(path, mode='a', maxBytes=setting_dict['logger']['max_bytes'],
                                     backupCount=setting_dict['logger']['backup_count'])
  file_handler.setFormatter(logging.Formatter(pattern))
  logger.addHandler(file_handler)

  logger.info(f'Logger initialized with {logging.getLevelName(logger.level)} level.')
