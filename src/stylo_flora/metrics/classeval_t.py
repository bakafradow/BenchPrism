import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from tqdm import tqdm

from .. import Snippet, setting_dict
from ..logger import logger
from . import utils
from .utils import Correctness

POM_PATH = (Path(setting_dict['datasets']['classeval_t_root']) /
            'BatchTestTool/java/java automated tester/pom.xml')


class CorrectnessCET(NamedTuple):
  corr: Correctness = Correctness.FAIL_COMP
  passed: int = 0
  total: int = 0


class CorrectnessResultCET(NamedTuple):
  comp_rate: float
  pass_rate_method: float
  pass_rate_class: float


PATTERN_MAVEN_TEST = re.compile(r'@Test')
PATTERN_MAVEN_STAT = re.compile(r'Tests run: (\d+), Failures: (\d+), Errors: (\d+)')


def checker_classeval(snippet: Snippet, lang: str) -> bool:
  result = pass_at_1_classeval([snippet.data['code']], [snippet.data[f'test_{lang}']], lang)
  return bool(result.pass_rate_class)


def pass_at_1_classeval(
    code_list: Seq[str],
    test_list: Seq[str],
    lang: str
) -> CorrectnessResultCET:
  initializer = getattr(sys.modules[__name__], f'initialize_{lang}', None)
  if initializer:
    initializer()
  tester = getattr(sys.modules[__name__], f'test_classeval_{lang}', None)
  if not tester:
    raise ValueError(f'Unsupported language: {lang}')
  with ThreadPoolExecutor(max_workers=setting_dict['metrics']['max_workers']) as executor:
    tuples = list(tqdm(executor.map(tester, code_list, test_list), desc='Calculating Pass@1',
                       total=len(code_list), leave=False))
  counter = Counter([t.corr for t in tuples])
  total = sum(counter.values())
  return CorrectnessResultCET(
      comp_rate=(total - counter[Correctness.FAIL_COMP]) / total,
      pass_rate_method=sum([t.passed for t in tuples]) / sum([t.total for t in tuples]),
      pass_rate_class=counter[Correctness.PASS] / total,
  )


@cache
def initialize_cpp() -> None:
  logger.info('Initializing C++ testing environment for ClassEval-T.')
  with open(f'{utils.get_msys_tmpdir()}/common.h', 'w') as f:
    f.write('#include <bits/stdc++.h>\n#include <sqlite3.h>\n')
  cmd_pch = ['g++', '-std=c++20', '-x', 'c++-header', f'{utils.get_msys_tmpdir()}/common.h',
             '-o', f'{utils.get_msys_tmpdir()}/common.h.pch']
  try:
    subprocess.run(cmd_pch, cwd=utils.get_msys_tmpdir_abs(), capture_output=True, check=True,
                   encoding='utf-8', errors='replace')
  except subprocess.CalledProcessError as e:
    raise ValueError(f'Failed to compile common.h:\n{e.stderr}')


def test_classeval_java(code: str, test: str) -> CorrectnessCET:
  classname = utils.extract_classname_java(code)
  total = len(PATTERN_MAVEN_TEST.findall(test))
  test_classes = utils.PATTERN_JAVA_CLASS.findall(test)
  if not test_classes:
    logger.verbose('No test classes found in the test code.')
    return CorrectnessCET(total=total)
  with tempfile.TemporaryDirectory(dir=utils.get_windows_tmpdir()) as tmpdir:
    shutil.copy(POM_PATH, f'{tmpdir}/pom.xml')
    os.makedirs(f'{tmpdir}/src/main/java', exist_ok=True)
    os.makedirs(f'{tmpdir}/src/test/java', exist_ok=True)
    with open(f'{tmpdir}/src/main/java/{classname}.java', 'w') as f:
      f.write(code)
    with open(f'{tmpdir}/src/test/java/{classname}Test.java', 'w') as f:
      f.write(test)
    cmd_compile = ['cmd.exe', '/c', 'javac.exe', '-d', f'{tmpdir}/target/classes',
                   f'{tmpdir}/src/main/java/{classname}.java']
    try:
      subprocess.run(cmd_compile, cwd=tmpdir, capture_output=True, check=True,
                     encoding='gbk', errors='replace')
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Failed to compile {classname}:\n{e.stderr}')
      return CorrectnessCET(total=total)
    cmd_test = ['cmd.exe', '/c', 'mvn.cmd', '-B', '-Dmaven.compiler.showWarnings=false',
                'test', f'-Dtest={",".join(test_classes)}']
    try:
      completed = subprocess.run(cmd_test, cwd=tmpdir, capture_output=True, check=True,
                                 encoding='gbk', errors='replace',
                                 timeout=setting_dict['metrics']['timeout'])
      stdout = completed.stdout
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Failed to test {classname}:\n{e.stderr}')
      stdout = e.stdout
    except subprocess.TimeoutExpired as e:
      logger.verbose(f'Test of {classname} timed out.')
      stdout = e.stdout  # type: ignore[assignment]
  if not stdout:
    logger.warning(f'Failed to get Maven output for {classname}.')
    return CorrectnessCET(Correctness.FAIL_EXEC, total=total)
  matched = PATTERN_MAVEN_STAT.search(stdout)
  if not matched:
    logger.warning(f'Failed to parse test result for {classname}.')
    return CorrectnessCET(Correctness.FAIL_EXEC, total=total)
  fails = int(matched.group(2))
  errors = int(matched.group(3))
  passed = total - fails - errors
  corr = Correctness.PASS if passed == total else Correctness.FAIL_EXEC
  return CorrectnessCET(corr, passed, total)


PATTERN_GTEST_TEST = re.compile(r'TEST(?:_F)?\s*\(')
PATTERN_GTEST_TOTAL = re.compile(r'\[={10}\] (\d+) tests? from')
PATTERN_GTEST_PASS = re.compile(r'\[ {2}PASSED {2}\] (\d+) tests?')


def test_classeval_cpp(code: str, test: str) -> CorrectnessCET:
  with tempfile.TemporaryDirectory(dir=utils.get_msys_tmpdir_abs()) as tmpdir:
    with open(f'{tmpdir}/pch.h', 'w') as f:
      f.write(code)
    if 'namespace example' in code:
      test = test.replace('#include "pch.h"', '# include "pch.h"\nusing namespace org::example;')
    with open(f'{tmpdir}/test.cpp', 'w') as f:
      f.write(test)
    total = len(PATTERN_GTEST_TEST.findall(test))
    tmpdir_rel = os.path.relpath(tmpdir, utils.get_msys_root())
    cmd_compile = ['g++', f'/{tmpdir_rel}/test.cpp', '-o', f'/{tmpdir_rel}/test.exe',
                   '-std=c++20', '-fuse-ld=lld',  # lld is slightly faster than default ld
                   '-include', f'{utils.get_msys_tmpdir()}/common.h',
                   '-lOpenXLSX', '-lboost_filesystem-mt', '-lgmock', '-lgtest',
                   '-ltinyxml', '-lsqlite3', '-lws2_32', '-lzip']
    try:
      utils.run_msys(cmd_compile, cwd=tmpdir, capture_output=True, check=True,
                     encoding='utf-8', errors='replace')
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Compilation Error:\n{e.stderr}')
      return CorrectnessCET(total=total)
    cmd_exe = [f'/{tmpdir_rel}/test.exe']
    try:
      completed = utils.run_msys(cmd_exe, cwd=tmpdir, capture_output=True,
                                 check=True, encoding='utf-8', errors='replace',
                                 timeout=setting_dict['metrics']['timeout'])
      stdout = completed.stdout
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Test failed:\n{e.stdout}')
      stdout = e.stdout
    except subprocess.TimeoutExpired as e:
      logger.verbose('Test timed out.')
      stdout = e.stdout  # type: ignore[assignment]
  if not stdout:
    logger.warning('Failed to get GTest output.')
    return CorrectnessCET(total=total)
  matched_total = PATTERN_GTEST_TOTAL.search(stdout)
  if not matched_total:
    logger.warning('Failed to parse test result.')
    return CorrectnessCET(Correctness.FAIL_EXEC, total=total)
  total = int(matched_total.group(1))
  matched_pass = PATTERN_GTEST_PASS.search(stdout)
  passed = int(matched_pass.group(1)) if matched_pass else 0
  corr = Correctness.PASS if passed == total else Correctness.FAIL_EXEC
  return CorrectnessCET(corr, passed, total)


PATTERN_UNITTEST_TEST = re.compile(r'^\s*def\s+test', re.M)
PATTERN_UNITTEST_TOTAL = re.compile(r'Ran (\d+) tests?')
PATTERN_UNITTEST_FAIL = re.compile(r'failures=(\d+)')
PATTERN_UNITTEST_ERROR = re.compile(r'errors=(\d+)')


def test_classeval_python(code: str, test: str) -> CorrectnessCET:
  """
  :note: Python environment requirements on MSYS2:
  - gensim~=4.2.0
  - numpy~=1.22.4
  - openpyxl~=3.0.9
  - pandas~=1.4.2
  - python~=3.10
  - scipy~=1.8.1
  """
  total = len(PATTERN_UNITTEST_TEST.findall(test))
  try:
    compile(code, '<string>', 'exec')
  except SyntaxError as e:
    logger.verbose(f'Syntax error:\n{e.msg}')
    return CorrectnessCET(total=total)
  matched = utils.PATTERN_PYTHON_CLASS.search(code)
  if not matched:
    logger.verbose('Class name not found.')
    return CorrectnessCET(total=total)
  module_name = f'{matched.group(1)}_{uuid4().hex}'
  with tempfile.TemporaryDirectory(dir=utils.get_msys_tmpdir_abs()) as tmpdir:
    path = os.path.join(tmpdir, f'{module_name}.py')
    with open(path, 'w') as f:
      f.write(f'{code}\n{test}')
    cmd = ['python', '-m', 'unittest', '-bfq', module_name]
    try:
      completed = utils.run_msys(cmd, cwd=tmpdir, capture_output=True, check=True,
                                 encoding='utf-8', errors='replace',
                                 timeout=setting_dict['metrics']['timeout'])
      stderr = completed.stderr
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Test failed for {module_name}:\n{e.stderr}')
      stderr = e.stderr
    except subprocess.TimeoutExpired as e:
      logger.verbose(f'Test timed out for {module_name}.')
      stderr = e.stderr  # type: ignore[assignment]
      cmd_kill = ['powershell.exe', '-NoProfile', '-Command',
                  f'Get-CimInstance Win32_Process | '
                  f'Where-Object {{ $_.CommandLine -like "*{module_name}*" -and '
                  f'$_.Name -like "python*" }} | '
                  f'Invoke-CimMethod -MethodName Terminate']
      completed = subprocess.run(cmd_kill, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, encoding='utf-8')
      if completed.returncode != 0:
        logger.warning(f'Failed to kill Python process for {module_name}.')
  if not stderr:
    logger.warning(f'Failed to get unittest output for {module_name}.')
    return CorrectnessCET(total=total)
  matched_total = PATTERN_UNITTEST_TOTAL.search(stderr)
  if not matched_total:
    logger.warning(f'Failed to parse test result for {module_name}.')
    return CorrectnessCET(total=total)
  total = int(matched_total.group(1))
  if 'OK' in stderr:
    return CorrectnessCET(Correctness.PASS, total, total)
  matched_fail = PATTERN_UNITTEST_FAIL.search(stderr)
  fails = int(matched_fail.group(1)) if matched_fail else 0
  matched_error = PATTERN_UNITTEST_ERROR.search(stderr)
  errors = int(matched_error.group(1)) if matched_error else 0
  passed = total - fails - errors
  return CorrectnessCET(Correctness.FAIL_EXEC, passed, total)
