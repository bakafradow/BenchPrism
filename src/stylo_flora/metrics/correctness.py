import os
import re
import shutil
import subprocess
import tempfile
import types
import unittest
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm

from .. import Snippet, TestBatch
from ..logger import logger

with open('settings.yml') as f:
  config = yaml.safe_load(f)['metrics']
TARGET_DIR = Path(config['target_dir'])
os.makedirs(TARGET_DIR, exist_ok=True)


class CompilationError(Exception):
  def __init__(self, id: str, message: str, stderr: str = ''):
    self.stderr = stderr
    super().__init__(f'{id}: {message}')


def _run_with_io(args: Sequence[str], tests: TestBatch, id_: str) -> bool:
  for input_, outputs in tqdm(tests, desc='Running tests', total=len(tests), leave=False):
    try:
      returned = subprocess.run(args, input=input_, text=True, capture_output=True, encoding='utf-8', timeout=config['timeout'])
    except KeyboardInterrupt:
      logger.warning('Keyboard interrupt.')
      raise
    except subprocess.TimeoutExpired:
      logger.verbose(f'Time out on {id_}.\n'
                     f'Input:\n{input_.strip()}')
      return False
    except Exception as e:
      logger.verbose(f'Error on {id_}.\n'
                     f'Input:\n{input_.strip()}\n'
                     f'Exception:\n{e}')
      return False
    if returned.returncode != 0:
      logger.verbose(f'{returned.returncode} was returned on {id_}.\n'
                     f'Input:\n{input_.strip()}\n'
                     f'Standard Error:\n{returned.stderr}')
      return False
    if returned.stdout.strip() not in (output.strip() for output in outputs):
      logger.verbose(f'Wrong answer on {id_}.\n'
                     f'Input:\n{input_.strip()}\n'
                     f'Expected:\n{outputs[0]}\n'
                     f'Actual:\n{returned.stdout}')
      return False
  return True


def _extrace_java_classname(snippet: Snippet) -> str:
  matched = re.search(r'public\s+(?:final\s+)?class\s+(\w+)', snippet.code)
  if not matched:
    matched = re.search(r'(?:final\s+)?class\s+(\w+)', snippet.code)
  if not matched:
    raise CompilationError(snippet.id, 'Failed to extract class name from Java code.', f'Generated code:\n{snippet.code}')
  return matched.group(1)


def test_java(snippet: Snippet) -> bool:
  try:
    classname = _extrace_java_classname(snippet)
    with tempfile.TemporaryDirectory() as tmpdir, open(f'{tmpdir}/{classname}.java', 'w') as f:
      f.write(snippet.code)
      classdir = tempfile.mkdtemp(dir=TARGET_DIR)
      try:
        returned = subprocess.run(['javac', '-d', classdir, f.name], stderr=subprocess.PIPE, encoding='utf-8', timeout=config['timeout'])
      except subprocess.TimeoutExpired:
        raise CompilationError(snippet.id, f'Compilation of {f.name} timed out.')
      if returned.returncode != 0:
        raise CompilationError(snippet.id, f'Failed to compile {f.name}.', returned.stderr)
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
    return False
  args = ['java', '-classpath', classdir, classname]
  result = _run_with_io(args, snippet.args['testcases'], snippet.id)
  shutil.rmtree(classdir, ignore_errors=True)
  return result


def test_java_repo(snippet: Snippet) -> bool:
  try:
    classname = _extrace_java_classname(snippet)
    test_classes = re.findall(r'class\s+(\w+)', snippet.args['test_code_java'])
    if not test_classes:
      raise CompilationError(snippet.id, 'No test classes found in the test code.')
    with tempfile.TemporaryDirectory() as tmpdir:
      os.makedirs(f'{tmpdir}/src/main/java', exist_ok=True)
      os.makedirs(f'{tmpdir}/src/test/java', exist_ok=True)
      with open(f'{tmpdir}/src/main/java/{classname}.java', 'w') as f:
        f.write(snippet.code)
      with open(f'{tmpdir}/src/test/java/{classname}Test.java', 'w') as f:
        f.write(snippet.args['test_code_java'])
      os.link('resources/pom.xml', f'{tmpdir}/pom.xml')
      returned = subprocess.run(['mvn', 'test', f'-Dtest={",".join(test_classes)}'],
                                cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding='utf-8', timeout=config['timeout'])
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
    logger.warning(f'Failed to build and test {classname}.\n'
                   f'Standard Error:\n{returned.stderr}')
    return False
  return True


def test_cpp(snippet: Snippet) -> bool:
  try:
    with tempfile.NamedTemporaryFile(suffix='.cpp') as f:
      f.write(snippet.code.encode())
      executable = re.sub(r'\.cpp$', '', f.name)
      try:
        returned = subprocess.run(['g++', f.name, '-o', executable], stderr=subprocess.PIPE, encoding='utf-8', timeout=config['timeout'])
      except subprocess.TimeoutExpired:
        raise CompilationError(snippet.id, f'Compilation of {f.name} timed out.')
      if returned.returncode != 0:
        raise CompilationError(snippet.id, f'Failed to compile {f.name}.', returned.stderr)
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
    return False
  args = [executable]
  result = _run_with_io(args, snippet.args['testcases'], snippet.id)
  os.remove(executable)
  return result


def test_cpp_repo(snippet: Snippet) -> bool:
  try:
    with tempfile.TemporaryDirectory() as tmpdir:
      with open(f'{tmpdir}/pch.h', 'w') as f:
        f.write(snippet.code)
      with open(f'{tmpdir}/test.cpp', 'w') as f:
        f.write(snippet.args['test_code_cpp'])
      os.link('resources/CMakeLists.txt', f'{tmpdir}/CMakeLists.txt')
      os.makedirs(f'{tmpdir}/build', exist_ok=True)
      returned = subprocess.run(['cmake', '..'], cwd=f'{tmpdir}/build', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', timeout=config['timeout'])
      if returned.returncode != 0:
        raise CompilationError(snippet.id, 'CMake configuration failed.', returned.stderr)
      returned = subprocess.run(['cmake', '--build', '.'], cwd=f'{tmpdir}/build', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', timeout=config['timeout'])
      if returned.returncode != 0:
        raise CompilationError(snippet.id, 'Building failed.', returned.stderr)
      returned = subprocess.run(['./test'], cwd=f'{tmpdir}/build', stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', timeout=config['timeout'])
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
    return False
  except subprocess.TimeoutExpired:
    logger.warning(f'Building and Testing of {snippet.id} timed out.')
    return False
  except Exception as e:
    logger.warning(f'Error during building and testing of {snippet.id}: {e}')
    return False
  if returned.returncode != 0:
    logger.warning(f'Failed to build and test {snippet.id}.\n'
                   f'Standard Error:\n{returned.stderr}')
    return False
  return True


def test_python(snippet: Snippet) -> bool:
  args = ['python', '-c', snippet.code]
  return _run_with_io(args, snippet.args['testcases'], snippet.id)


def test_python_repo(snippet: Snippet) -> bool:
  module = types.ModuleType(snippet.id)
  exec(snippet.code, module.__dict__)
  exec(snippet.args['test_code_python'], module.__dict__)
  loader = unittest.TestLoader()
  suite = loader.loadTestsFromModule(module)
  runner = unittest.TextTestRunner(verbosity=0, failfast=True)
  result = runner.run(suite)
  return result.wasSuccessful()


# TODO: 1. execute in Docker
#       2. change to pass@k
def calc_correctness(snippets: Sequence[Snippet], lang: str) -> float:
  """
  Checks the correctness of the translated code with the tests.
  :param snippets: the translated code snippets
  :param lang: the language of the code snippets
  """
  def worker(snippet: Snippet) -> bool:
    if not snippet:
      return False
    try:
      if snippet.args.get(f'test_code_{lang}'):
        return globals()[f'test_{lang}_repo'](snippet)
      return globals()[f'test_{lang}'](snippet)
    except KeyError:
      raise TypeError(f'Unsupported language {lang} for correctness testing.')

  max_workers = max(1, config['max_workers'])
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(worker, snippets), desc='Calculating correctness', total=len(snippets), leave=False))
  return sum(results) / len(snippets)
