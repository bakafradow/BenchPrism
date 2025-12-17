import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from tqdm import tqdm

from .. import IOTestCase, setting_dict
from ..logger import logger
from .utils import Correctness, extract_classname_java


class CorrectnessResult(NamedTuple):
  comp_rate: float
  pass_rate: float


def pass_at_1(code_list: Seq[str], tc_lists: Seq[Seq[IOTestCase]], lang: str) -> CorrectnessResult:
  tester = getattr(sys.modules[__name__], f'test_io_{lang}', None)
  if not tester:
    raise ValueError(f'Unsupported language: {lang}')
  with ThreadPoolExecutor(max_workers=setting_dict['metrics']['max_workers']) as executor:
    counter = Counter(tqdm(executor.map(tester, code_list, tc_lists),
                           desc='Calculating Pass@1', total=len(code_list), leave=False))
  total = sum(counter.values())
  return CorrectnessResult(
      comp_rate=(total - counter[Correctness.FAIL_COMP]) / total,
      pass_rate=counter[Correctness.PASS] / total,
  )


def test_io_java(code: str, tc_list: Seq[IOTestCase]) -> Correctness:
  classname = extract_classname_java(code)
  if not classname:
    logger.verbose('Failed to extract class name from code.')
    return Correctness.FAIL_COMP
  with tempfile.TemporaryDirectory() as tmpdir:
    with open(os.path.join(tmpdir, f'{classname}.java'), 'w') as f:
      f.write(code)
    classdir = os.path.join(tmpdir, 'target')
    os.makedirs(classdir, exist_ok=True)
    cmd_compile = ['javac', '-d', classdir, f.name]
    try:
      subprocess.run(cmd_compile, capture_output=True, check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Failed to compile {f.name}:\n{e.stderr}')
      return Correctness.FAIL_COMP
    cmd = ['java', '-classpath', classdir, classname]
    return _run_with_io(cmd, tc_list, tmpdir)


def test_io_cpp(code: str, tc_list: Seq[IOTestCase]) -> Correctness:
  with tempfile.TemporaryDirectory() as tmpdir:
    src_path = os.path.join(tmpdir, 'main.cpp')
    exe_path = os.path.join(tmpdir, 'main')
    with open(src_path, 'w') as f:
      f.write(code)
    cmd_compile = ['g++', src_path, '-o', exe_path]
    try:
      subprocess.run(cmd_compile, capture_output=True, check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Failed to compile:\n{e.stderr}')
      return Correctness.FAIL_COMP
    cmd = [exe_path]
    return _run_with_io(cmd, tc_list, tmpdir)


def test_io_python(code: str, tc_list: Seq[IOTestCase]) -> Correctness:
  try:
    compile(code, '<string>', 'exec')
  except SyntaxError as e:
    logger.verbose(f'Syntax error:\n{e.msg}')
    return Correctness.FAIL_COMP
  with tempfile.TemporaryDirectory() as tmpdir:
    src_path = os.path.join(tmpdir, 'main.py')
    with open(src_path, 'w') as f:
      f.write(code)
    cmd = ['python', src_path]
    return _run_with_io(cmd, tc_list, tmpdir)


def _run_with_io(cmd: Seq[str], tc_list: Seq[IOTestCase], cwd: str) -> Correctness:
  def worker(test: IOTestCase) -> bool:
    try:
      completed = subprocess.run(cmd, cwd=cwd, capture_output=True, encoding='utf-8',
                                 input=test.input, timeout=setting_dict['metrics']['timeout'])
      if completed.returncode != 0:
        logger.verbose(f'{completed.returncode} was returned.\n'
                       f'Input:\n{test.input.strip()}\n'
                       f'Standard Error:\n{completed.stderr}')
        return False
      if completed.stdout.strip() not in (output.strip() for output in test.outputs):
        logger.verbose(f'Wrong answer.\n'
                       f'Input:\n{test.input.strip()}\n'
                       f'Expected:\n{test.outputs[0]}\n'
                       f'Actual:\n{completed.stdout}')
        return False
    except KeyboardInterrupt:
      logger.warning('Keyboard interrupt.')
      raise
    except subprocess.TimeoutExpired:
      logger.verbose(f'Time out.\n'
                     f'Input:\n{test.input.strip()}')
      return False
    except Exception as e:
      logger.verbose(f'Error.\n'
                     f'Input:\n{test.input.strip()}\n'
                     f'Exception:\n{e}')
      return False
    return True

  return Correctness.PASS if all(map(worker, tc_list)) else Correctness.FAIL_EXEC
