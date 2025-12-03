import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from .. import IOTestCase, setting_dict
from ..logger import logger
from .utils import CompilationError, extract_classname_java


def _run_with_io(cmd: Seq[str], tc_list: Seq[IOTestCase]) -> bool:
  def worker(test: IOTestCase) -> bool:
    try:
      completed = subprocess.run(cmd, input=test.input, text=True, capture_output=True, encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
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
    return True

  with ThreadPoolExecutor(max_workers=setting_dict['metrics']['max_workers']) as executor:
    return all(tqdm(executor.map(worker, tc_list), total=len(tc_list), leave=False))


def test_io_java(code: str, tc_list: Seq[IOTestCase]) -> bool:
  try:
    classname = extract_classname_java(code)
    if not classname:
      raise CompilationError('Failed to extract class name from Java code.', f'Generated code:\n{code}')
    with tempfile.TemporaryDirectory() as tmpdir:
      with open(f'{tmpdir}/{classname}.java', 'w') as f:
        f.write(code)
      classdir = f'{tmpdir}/target'
      os.makedirs(classdir, exist_ok=True)
      try:
        completed = subprocess.run(['javac', '-d', classdir, f.name], stderr=subprocess.PIPE, encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
      except subprocess.TimeoutExpired:
        raise CompilationError(f'Compilation of {f.name} timed out.')
      if completed.returncode != 0:
        raise CompilationError(f'Failed to compile {f.name}.', completed.stderr)
      cmd = ['java', '-classpath', classdir, classname]
      return _run_with_io(cmd, tc_list)
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
  return False


def test_io_cpp(code: str, tc_list: Seq[IOTestCase]) -> bool:
  try:
    with tempfile.NamedTemporaryFile(suffix='.cpp') as f:
      f.write(code.encode())
      f.flush()
      executable = re.sub(r'\.cpp$', '', f.name)
      try:
        completed = subprocess.run(['g++', f.name, '-o', executable], stderr=subprocess.PIPE, encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
      except subprocess.TimeoutExpired:
        raise CompilationError(f'Compilation of {f.name} timed out.')
      if completed.returncode != 0:
        raise CompilationError(f'Failed to compile {f.name}.', completed.stderr)
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
    return False
  cmd = [executable]
  result = _run_with_io(cmd, tc_list)
  try:
    os.remove(executable)
  except FileNotFoundError:
    pass
  return result


def test_io_python(code: str, tc_list: Seq[IOTestCase]) -> bool:
  cmd = ['python', '-c', code]
  return _run_with_io(cmd, tc_list)


def pass_at_1(code_list: Seq[str], tc_lists: Seq[Seq[IOTestCase]], lang: str) -> float:
  tester = getattr(sys.modules[__name__], f'test_io_{lang}', None)
  if not tester:
    raise ValueError(f'Unsupported language: {lang}')
  return sum(tester(code, tc_list)
             for code, tc_list in tqdm(zip(code_list, tc_lists),
                                       desc='Calculating Pass@1', total=len(code_list),
                                       leave=False)) / len(code_list)
