import os
import re
import subprocess
from enum import Enum, auto
from functools import cache
from typing import Any

from ..logger import logger


class Correctness(str, Enum):
  @staticmethod
  def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> Any:
    return name

  FAIL_COMP = auto()
  FAIL_EXEC = auto()
  PASS = auto()


PATTERN_JAVA_PUBLIC_CLASS = re.compile(r'public\s+(?:final\s+)?class\s+(\w+)')
PATTERN_JAVA_CLASS = re.compile(r'(?:final\s+)?class\s+(\w+)')
PATTERN_PYTHON_CLASS = re.compile(r'class\s+(\w+)')


def extract_classname_java(code: str) -> str | None:
  matched = PATTERN_JAVA_PUBLIC_CLASS.search(code)
  if not matched:
    matched = PATTERN_JAVA_CLASS.search(code)
  if not matched:
    return None
  return matched.group(1)


@cache
def get_windows_tmpdir() -> str:
  try:
    cmd_echo = ['cmd.exe', '/c', 'echo', '%TEMP%']
    completed = subprocess.run(cmd_echo, capture_output=True, check=True, encoding='utf-8')
    path = completed.stdout.strip()
    cmd_wslpath = ['wslpath', '-u', path]
    completed = subprocess.run(cmd_wslpath, capture_output=True, check=True, encoding='utf-8')
    tmpdir = completed.stdout.strip()
    logger.verbose(f'Using Windows temp directory: {tmpdir}')
    return tmpdir
  except subprocess.CalledProcessError as e:
    logger.warning(f'Failed to get Windows temp directory:\n{e.stderr}')
    raise


@cache
def get_msys_root() -> str:
  msys_root = os.getenv('MSYS2_ROOT', '/mnt/c/msys64')
  if not os.path.exists(msys_root):
    raise ValueError('MSYS2 UCRT64 on Windows is required for ClassEval-T C++ testing. '
                     'Please set MSYS2_ROOT environment variable correctly.')
  logger.verbose(f'Using MSYS2 root: {msys_root}')
  return msys_root


def run_msys(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
  cmd_in_bash = (
      'export MSYSTEM=UCRT64 && '
      'export CHERE_INVOCATION=1 && '
      'CWD="$(pwd)" && '
      'source /etc/profile && '
      'cd "$CWD" && '
      'unset CWD && '
      f'exec {" ".join(cmd)}'
  )
  cmd = [os.path.join(get_msys_root(), 'usr', 'bin', 'bash.exe'), '-c', cmd_in_bash]
  return subprocess.run(cmd, **kwargs)


@cache
def get_msys_tmpdir() -> str:
  try:
    cmd = ['echo', '$TEMP']
    completed = run_msys(cmd, capture_output=True, check=True, encoding='utf-8', errors='replace')
    tmpdir = completed.stdout.strip()
    logger.verbose(f'Using MSYS2 temp directory: {tmpdir}')
    return tmpdir
  except subprocess.CalledProcessError as e:
    logger.warning(f'Failed to get MSYS2 temp directory:\n{e.stderr}')
    raise


@cache
def get_msys_tmpdir_abs() -> str:
  return get_msys_root() + get_msys_tmpdir()
