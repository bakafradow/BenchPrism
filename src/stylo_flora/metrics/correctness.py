import os
import re
import shutil
import subprocess
import tempfile
import types
import unittest
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm import tqdm

from .. import IOTestCase, setting_dict
from ..logger import logger
from .utils import extract_classname_java


class CompilationError(Exception):
  def __init__(self, message: str, stderr: str = ''):
    self.stderr = stderr
    super().__init__(message)


def _run_with_io(cmd: Seq[str], tests: Seq[IOTestCase]) -> bool:
  for test in tqdm(tests, desc='Running tests', total=len(tests), leave=False):
    try:
      returned = subprocess.run(cmd, input=test.input, text=True, capture_output=True, encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
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
    if returned.returncode != 0:
      logger.verbose(f'{returned.returncode} was returned.\n'
                     f'Input:\n{test.input.strip()}\n'
                     f'Standard Error:\n{returned.stderr}')
      return False
    if returned.stdout.strip() not in (output.strip() for output in test.outputs):
      logger.verbose(f'Wrong answer.\n'
                     f'Input:\n{test.input.strip()}\n'
                     f'Expected:\n{test.outputs[0]}\n'
                     f'Actual:\n{returned.stdout}')
      return False
  return True


def test_io_java(code: str, args: dict) -> bool:
  classname = extract_classname_java(code)
  if not classname:
    raise CompilationError('Failed to extract class name from Java code.', f'Generated code:\n{code}')
  try:
    with tempfile.TemporaryDirectory() as tmpdir:
      with open(f'{tmpdir}/{classname}.java', 'w') as f:
        f.write(code)
      classdir = f'{tmpdir}/target'
      os.makedirs(classdir, exist_ok=True)
      try:
        returned = subprocess.run(['javac', '-d', classdir, f.name], stderr=subprocess.PIPE, encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
      except subprocess.TimeoutExpired:
        raise CompilationError(f'Compilation of {f.name} timed out.')
      if returned.returncode != 0:
        raise CompilationError(f'Failed to compile {f.name}.', returned.stderr)
      cmd = ['java', '-classpath', classdir, classname]
      return _run_with_io(cmd, args['io_testcases'])
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
  return False


def test_api_java(code: str, args: dict) -> bool:
  try:
    classname = extract_classname_java(code)
    test_classes = re.findall(r'class\s+(\w+)', args['api_testcases_java'].code)
    if not test_classes:
      raise CompilationError('No test classes found in the test code.')
    with tempfile.TemporaryDirectory() as tmpdir:
      os.makedirs(f'{tmpdir}/src/main/java', exist_ok=True)
      os.makedirs(f'{tmpdir}/src/test/java', exist_ok=True)
      with open(f'{tmpdir}/src/main/java/{classname}.java', 'w') as f:
        f.write(code)
      with open(f'{tmpdir}/src/test/java/{classname}Test.java', 'w') as f:
        f.write(args['api_testcases_java'].code)
      shutil.copy('resources/pom.xml', f'{tmpdir}/pom.xml')
      returned = subprocess.run(['mvn', 'test', f'-Dtest={",".join(test_classes)}'],
                                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
    return False
  except subprocess.TimeoutExpired:
    logger.warning(f'Building and Testing of {classname} timed out.')
    return False
  except Exception as e:
    logger.warning(f'Error during building and testing of {classname}: {e}')
    return False
  if returned.returncode != 0:
    logger.warning(f'Failed to build and test {classname}.')
    logger.verbose(f'Standard Output:\n{returned.stdout}')
    return False
  return True


def test_io_cpp(code: str, args: dict) -> bool:
  try:
    with tempfile.NamedTemporaryFile(suffix='.cpp') as f:
      f.write(code.encode())
      f.flush()
      executable = re.sub(r'\.cpp$', '', f.name)
      try:
        returned = subprocess.run(['g++', f.name, '-o', executable], stderr=subprocess.PIPE, encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
      except subprocess.TimeoutExpired:
        raise CompilationError(f'Compilation of {f.name} timed out.')
      if returned.returncode != 0:
        raise CompilationError(f'Failed to compile {f.name}.', returned.stderr)
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
    return False
  cmd = [executable]
  result = _run_with_io(cmd, args['io_testcases'])
  os.remove(executable)
  return result


def test_api_cpp(code: str, args: dict) -> bool:
  try:
    with tempfile.TemporaryDirectory() as tmpdir:
      with open(f'{tmpdir}/pch.h', 'w') as f:
        f.write(code)
      with open(f'{tmpdir}/test.cpp', 'w') as f:
        f.write(args['api_testcases_cpp'].code)
      shutil.copy('resources/CMakeLists.txt', f'{tmpdir}/CMakeLists.txt')
      os.makedirs(f'{tmpdir}/build', exist_ok=True)
      returned = subprocess.run(['cmake', '..'], cwd=f'{tmpdir}/build', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
      if returned.returncode != 0:
        raise CompilationError('CMake configuration failed.', returned.stderr)
      returned = subprocess.run(['cmake', '--build', '.'], cwd=f'{tmpdir}/build', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
      if returned.returncode != 0:
        raise CompilationError('Building failed.', returned.stderr)
      returned = subprocess.run(['./test'], cwd=f'{tmpdir}/build', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
    return False
  except subprocess.TimeoutExpired:
    logger.warning('Building and Testing timed out.')
    return False
  except Exception as e:
    logger.warning(f'Error during building and testing: {e}')
    return False
  if returned.returncode != 0:
    logger.warning('Failed to build and test.')
    logger.verbose(f'Standard Output:\n{returned.stdout}')
    return False
  return True


def test_io_python(code: str, args: dict) -> bool:
  cmd = ['python', '-c', code]
  return _run_with_io(cmd, args['io_testcases'])


def test_api_python(code: str, args: dict) -> bool:
  module = types.ModuleType('focal_module')
  exec(code, module.__dict__)
  exec(args['api_testcases_python'].code, module.__dict__)
  loader = unittest.TestLoader()
  suite = loader.loadTestsFromModule(module)
  runner = unittest.TextTestRunner(verbosity=0, failfast=True)
  result = runner.run(suite)
  return result.wasSuccessful()


def calc_correctness(code_list: Seq[str], args_list: Seq[dict[str, Any]], lang: str) -> float:
  """
  Checks the correctness of the translated code with the tests.
  :param snippets: the translated code snippets
  :param lang: the language of the code snippets
  """
  def worker(code: str, args: dict) -> bool:
    try:
      if args.get(f'api_testcases_{lang}'):
        return globals()[f'test_api_{lang}'](code, args)
      return globals()[f'test_io_{lang}'](code, args)
    except KeyError:
      raise TypeError(f'Unsupported language {lang} for correctness testing.')

  max_workers = max(1, setting_dict['metrics']['max_workers'])
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(worker, code_list, args_list),
                        desc='Calculating correctness', total=len(code_list), leave=False))
  return sum(results) / len(code_list)
