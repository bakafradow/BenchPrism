import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from functools import cache

from tqdm import tqdm

from .. import setting_dict
from ..logger import logger
from . import utils
from .utils import CompilationError


def test_classeval_java(code: str, test: str) -> bool:
  try:
    classname = utils.extract_classname_java(code)
    test_classes = re.findall(r'class\s+(\w+)', test)
    if not test_classes:
      raise CompilationError('No test classes found in the test code.')
    with tempfile.TemporaryDirectory(dir=utils.get_windows_tmpdir()) as tmpdir:
      shutil.copy('data/ClassEval-T/BatchTestTool/java/java automated tester/pom.xml',
                  f'{tmpdir}/pom.xml')
      os.makedirs(f'{tmpdir}/src/main/java', exist_ok=True)
      os.makedirs(f'{tmpdir}/src/test/java', exist_ok=True)
      with open(f'{tmpdir}/src/main/java/{classname}.java', 'w') as f:
        f.write(code)
      with open(f'{tmpdir}/src/test/java/{classname}Test.java', 'w') as f:
        f.write(test)
      cmd = ['cmd.exe', '/c', 'mvn.cmd', 'test', f'-Dtest={",".join(test_classes)}']
      completed = subprocess.run(cmd, cwd=tmpdir, encoding='gbk', errors='replace',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=setting_dict['metrics']['timeout'])
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
    return False
  except subprocess.TimeoutExpired:
    logger.warning(f'Testing of {classname} timed out.')
    return False
  except Exception as e:
    logger.warning(f'{e.__class__.__name__} occurred during testing of {classname}:\n{e}')
    return False
  if completed.returncode != 0:
    logger.warning(f'Failed to test {classname}.')
    logger.verbose(f'Standard Output:\n{completed.stdout}')
    return False
  return True


def test_classeval_cpp(code: str, test: str) -> bool:
  try:
    with tempfile.TemporaryDirectory(dir=utils.get_msys_tmpdir_abs()) as tmpdir:
      with open(f'{tmpdir}/pch.h', 'w') as f:
        f.write(code)
      if 'namespace example' in code:
        test = test.replace('#include "pch.h"', '# include "pch.h"\nusing namespace org::example;')
      with open(f'{tmpdir}/test.cpp', 'w') as f:
        f.write(test)
      tmpdir_rel = os.path.relpath(tmpdir, utils.get_msys_root())
      cmd_compile = ['g++', f'/{tmpdir_rel}/test.cpp', '-o', f'/{tmpdir_rel}/test.exe',
                     '-std=c++20', '-fuse-ld=lld',  # lld is slightly faster than default ld
                     '-include', f'{utils.get_msys_tmpdir()}/common.h',
                     '-lOpenXLSX', '-lboost_filesystem-mt', '-lgmock', '-lgtest',
                     '-ltinyxml', '-lsqlite3', '-lws2_32', '-lzip']
      completed = utils.run_msys(cmd_compile, cwd=tmpdir, encoding='utf-8', errors='replace',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=setting_dict['metrics']['timeout'])
      if completed.returncode != 0:
        raise CompilationError('Compilation failed.', completed.stderr)
      cmd_exe = [f'/{tmpdir_rel}/test.exe']
      completed = utils.run_msys(cmd_exe, cwd=tmpdir, encoding='utf-8', errors='replace',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=setting_dict['metrics']['timeout'])
  except CompilationError as e:
    logger.warning(e)
    logger.verbose(f'Standard Error:\n{e.stderr}')
    return False
  except subprocess.TimeoutExpired:
    logger.warning('Testing timed out.')
    return False
  except Exception as e:
    logger.warning(f'{e.__class__.__name__} occurred during testing: {e}')
    return False
  if completed.returncode != 0:
    logger.warning(f'Test failed with exit code {completed.returncode}.')
    logger.verbose(f'Standard Output:\n{completed.stdout}')
    return False
  return True


def test_classeval_python(code: str, test: str) -> bool:
  """
  :note: Python environment requirements on MSYS2:
  - gensim~=4.2.0
  - numpy~=1.22.4
  - openpyxl~=3.0.9
  - pandas~=1.4.2
  - python~=3.10
  - scipy~=1.8.1
  """
  try:
    matched = re.search(r'class\s+(\w+)', code)
    if not matched:
      raise CompilationError('Class name not found.')
    module_name = matched.group(1)
    with tempfile.TemporaryDirectory(dir=utils.get_msys_tmpdir_abs()) as tmpdir:
      path = os.path.join(tmpdir, f'{module_name}.py')
      with open(path, 'w') as f:
        f.write(f'{code}\n{test}')
      cmd = ['cd', f'/{os.path.relpath(tmpdir, utils.get_msys_root())}', '&&',
             'python', '-m', 'unittest', '-bfq', module_name]
      completed = utils.run_msys(cmd, cwd=tmpdir, encoding='utf-8', errors='replace',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=setting_dict['metrics']['timeout'])
  except Exception as e:
    logger.warning(f'{e.__class__.__name__} occurred while testing:\n{e}')
    return False
  if completed.returncode != 0:
    logger.warning(f'Test failed with exit code {completed.returncode}.')
    logger.verbose(f'Standard Error:\n{completed.stderr}')
    return False
  return True


@cache
def initialize_cpp() -> None:
  logger.info('Initializing C++ testing environment for ClassEval-T.')
  with open(f'{utils.get_msys_tmpdir()}/common.h', 'w') as f:
    f.write('#include <bits/stdc++.h>\n#include <sqlite3.h>\n')
  cmd_pch = ['g++', '-std=c++20', '-x', 'c++-header', f'{utils.get_msys_tmpdir()}/common.h',
             '-o', f'{utils.get_msys_tmpdir()}/common.h.pch']
  completed = subprocess.run(cmd_pch, cwd=utils.get_msys_tmpdir_abs(), encoding='utf-8',
                             errors='replace', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if completed.returncode != 0:
    raise CompilationError(f'Failed to compile common.h:\n{completed.stderr}')


def pass_at_1_classeval(code_list: Seq[str], test_list: Seq[str], lang: str) -> float:
  initializer = getattr(sys.modules[__name__], f'initialize_{lang}', None)
  if initializer:
    initializer()
  tester = getattr(sys.modules[__name__], f'test_classeval_{lang}', None)
  if not tester:
    raise ValueError(f'Unsupported language: {lang}')
  with ThreadPoolExecutor(max_workers=setting_dict['metrics']['max_workers']) as executor:
    return sum(tqdm(executor.map(tester, code_list, test_list), desc='Calculating Pass@1',
                    total=len(code_list), leave=False)) / len(code_list)
