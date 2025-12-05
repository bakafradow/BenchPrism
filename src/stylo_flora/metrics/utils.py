import os
import re
import subprocess
from functools import cache

from ..logger import logger


class CompilationError(Exception):
  def __init__(self, message: str, stderr: str = ''):
    self.stderr = stderr
    super().__init__(message)


PATTERN_JAVA_PUBLIC_CLASS = re.compile(r'public\s+(?:final\s+)?class\s+(\w+)')
PATTERN_JAVA_CLASS = re.compile(r'(?:final\s+)?class\s+(\w+)')


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
    logger.warning(f'Failed to get Windows temp directory: {e}')
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
  cmd = [os.path.join(get_msys_root(), 'usr', 'bin', 'env.exe'),
         'MSYSTEM=UCRT64', '/usr/bin/bash', '-l', '-c', ' '.join(cmd)]
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
    logger.warning(f'Failed to get MSYS2 temp directory: {e}')
    raise


@cache
def get_msys_tmpdir_abs() -> str:
  return get_msys_root() + get_msys_tmpdir()
