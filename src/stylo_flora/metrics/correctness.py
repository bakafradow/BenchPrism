import os
import re
import shutil
import subprocess
import tempfile
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


def test_java(snippet: Snippet) -> bool:
  try:
    matched = re.search(r'public\s+(?:final\s+)?class\s+(\w+)', snippet.code)
    if not matched:
      matched = re.search(r'(?:final\s+)?class\s+(\w+)', snippet.code)
    if not matched:
      raise CompilationError(snippet.id, 'Failed to extract class name from Java code.', f'Generated code:\n{snippet.code}')
    classname = matched.group(1)
    with tempfile.TemporaryDirectory() as tmpdir, open(f'{tmpdir}/{classname}.java', 'w') as f:
      f.write(snippet.code)
      f.flush()
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
  args = ['java', '-classpath', f'{classdir}', classname]
  # TODO: run snippet.args['tester'] if available
  result = _run_with_io(args, snippet.args['testcases'], snippet.id)
  shutil.rmtree(classdir, ignore_errors=True)
  return result


def test_cpp(snippet: Snippet) -> bool:
  try:
    with tempfile.NamedTemporaryFile(suffix='.cpp') as f:
      f.write(snippet.code.encode())
      f.flush()
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


def test_python(snippet: Snippet) -> bool:
  args = ['python', '-c', snippet.code]
  return _run_with_io(args, snippet.args['testcases'], snippet.id)


# TODO: execute in Docker
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
      return globals()[f'test_{lang}'](snippet)
    except KeyError:
      raise TypeError(f'Unsupported language {lang} for correctness testing.')

  max_workers = max(1, config['max_workers'])
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(worker, snippets), desc='Calculating correctness', total=len(snippets), leave=False))
  return sum(results) / len(snippets)
