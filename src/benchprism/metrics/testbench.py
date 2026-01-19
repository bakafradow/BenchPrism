"""
Modified from TestBench repository (https://github.com/iSEngLab/TestBench).
"""

import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterator
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, NamedTuple

import numpy as np
from tqdm import tqdm

from .. import Snippet, setting_dict
from ..logger import logger
from .utils import Correctness, extract_classname_java

PROJ_PATH = Path(setting_dict['datasets']['testbench_root']) / 'java_project'


class CoverageResultTB(NamedTuple):
  comp_rate: float
  pass_rate: float
  line_cov: float
  branch_cov: float
  mut_score: float


PATTERN_FUNC = re.compile(r'class\s+[\w\$]+[^\{]+\{\n(.+)\n\}', re.S)
PATTERN_MAVEN_STAT = re.compile(r'Tests run: (\d+), Failures: (\d+), Errors: (\d+)')


def checker_testbench(snippet: Snippet, lang: str) -> bool:
  item = snippet.data
  identifier = f'{item["class_name"]}::{item["method_name"]}'

  with _git_worktree(PROJ_PATH / item['project_name']) as repo_root:
    matched = PATTERN_FUNC.search(item['code'])
    if not matched:
      logger.warning(f'Failed to unwrap function while checking {snippet.id}.')
      return False
    variant_code = matched.group(1)
    variant_src = _replace_code(item['full_context'], item['source_code'], variant_code)
    if not variant_src:
      logger.warning(f'Failed to replace function while checking {snippet.id}.')
      return False

    _, src_path_rel = item['relative_path'].split('/', 1)
    test_dir = repo_root / src_path_rel.replace('main', 'test').rsplit('/', 1)[0]
    src_path = repo_root / src_path_rel
    with open(src_path, 'w') as f:
      f.write(variant_src)

    mvn_prefix = ['mvn', '-B', '-Dmaven.compiler.showWarnings=false']
    submodule = _get_submodule(src_path_rel)
    if submodule:
      mvn_prefix += ['-pl', submodule, '-am']  # avoid building unnecessary modules
    test_pattern = str(test_dir).split('src/test/java/', 1)[-1].rstrip('/').replace('/', '.') + '.**'
    cmd_test = mvn_prefix + ['clean', 'test', '-DskipPitest=True', '-Drat.skip=true',
                             '-Dsurefire.failIfNoSpecifiedTests=false', '-Dcheckstyle.skip',
                             '-Dmaven.gitcommitid.skip', f'-Dtest={test_pattern}']
    try:
      completed = subprocess.run(cmd_test, cwd=repo_root, capture_output=True, check=True,
                                 encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
      matched = PATTERN_MAVEN_STAT.search(completed.stdout)
      if not matched:
        logger.verbose(f'{identifier} has no test result. Skipping.')
        return True
      fails = int(matched.group(2))
      errors = int(matched.group(3))
      if fails or errors:
        logger.verbose(f'Test of {identifier} has {fails} fails, {errors} errors.')
        return False
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Test of {identifier} failed.\n{e.stdout}')
      return False
    except subprocess.TimeoutExpired:
      logger.verbose(f'Test of {identifier} timed out.')
      return False
    return True


def _replace_code(full_context: str, source_code: str, variant_code: str) -> str | None:
  ctx_lines = full_context.splitlines(keepends=True)
  src_lines = source_code.strip().splitlines()

  start_idx = next(i for i in range(len(ctx_lines))
                   if ctx_lines[i].strip() == src_lines[0].strip() and
                   ctx_lines[i + 1].strip() == src_lines[1].strip())
  end_idx = start_idx + len(src_lines)

  first_line = ctx_lines[start_idx]
  indent = first_line[:first_line.find(src_lines[0].strip())]

  variant_lines = [indent + line + '\n' if line.strip() else '\n'
                   for line in variant_code.splitlines()]
  return ''.join(ctx_lines[:start_idx] + variant_lines + ctx_lines[end_idx:])


def calc_coverage_tb(tests: Seq[str], items: Seq[dict[str, Any]], lang: str) -> CoverageResultTB:
  if lang != 'java':
    raise ValueError(f'Unsupported language: {lang}')
  with ThreadPoolExecutor(max_workers=setting_dict['metrics']['max_workers']) as executor:
    dicts = list(tqdm(executor.map(_worker, tests, items),
                      desc='Calculating coverage', total=len(tests), leave=False))
  counter = Counter([d['correctness'] for d in dicts])
  total = sum(counter.values())
  return CoverageResultTB(
      comp_rate=(total - counter[Correctness.FAIL_COMP]) / total,
      pass_rate=counter[Correctness.PASS] / total,
      line_cov=np.mean([d.get('line_cov') or .0 for d in dicts]),
      branch_cov=np.mean([d.get('branch_cov') or .0 for d in dicts]),
      mut_score=np.mean([d.get('mut_score') or .0 for d in dicts]),
  )


def _worker(test: str, item: dict[str, Any]) -> dict[str, Any]:
  result: dict[str, Any] = {'correctness': Correctness.FAIL_COMP}
  identifier = f'{item["class_name"]}::{item["method_name"]}'
  classname = extract_classname_java(test)
  if not classname:
    logger.verbose(f'Failed to extract class name for {identifier} test.')
    return result

  with _git_worktree(PROJ_PATH / item['project_name']) as repo_root:
    _, src_path_rel = item['relative_path'].split('/', 1)
    test_dir = repo_root / src_path_rel.replace('main', 'test').rsplit('/', 1)[0]
    test_dir.mkdir(parents=True, exist_ok=True)
    with open(test_dir / f'{classname}.java', 'w') as f:
      f.write(test)

    mvn_prefix = ['mvn', '-B', '-Dmaven.compiler.showWarnings=false']
    submodule = _get_submodule(src_path_rel)
    if submodule:
      mvn_prefix += ['-pl', submodule, '-am']  # avoid building unnecessary modules
    cmd_compile = mvn_prefix + ['-q', 'clean', 'test-compile', '-Drat.skip=true',
                                '-Dsurefire.failIfNoSpecifiedTests=false', '-Dcheckstyle.skip',
                                '-Dmaven.gitcommitid.skip']
    try:
      subprocess.run(cmd_compile, cwd=repo_root, capture_output=True, check=True,
                     encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Failed to compile {identifier}\n{e.stdout}')
      return result
    except subprocess.TimeoutExpired:
      logger.verbose(f'Compilation of {identifier} timed out.')
      return result
    result['correctness'] = Correctness.FAIL_EXEC

    cmd_test = mvn_prefix + ['clean', 'test', '-DskipPitest=True', '-Drat.skip=true',
                             '-Dsurefire.failIfNoSpecifiedTests=false', '-Dcheckstyle.skip',
                             '-Dmaven.gitcommitid.skip', f'-Dtest={classname}']
    try:
      completed = subprocess.run(cmd_test, cwd=repo_root, capture_output=True, encoding='utf-8',
                                 timeout=setting_dict['metrics']['timeout'])
      matched = PATTERN_MAVEN_STAT.search(completed.stdout)
      if not matched:
        logger.warning(f'Failed to parse test result for {identifier}.')
        logger.debug(completed.stdout)
        return result
      fails = int(matched.group(2))
      errors = int(matched.group(3))
      if fails or errors:
        logger.verbose(f'Test of {identifier} has {fails} fails, {errors} errors.')
        return result
    except subprocess.TimeoutExpired:
      logger.verbose(f'Test of {identifier} timed out.')
      return result
    result['correctness'] = Correctness.PASS
    report_path = os.path.join(repo_root, submodule, 'target/site/jacoco/jacoco.xml')
    if not os.path.exists(report_path):
      logger.warning(f'Failed to find coverage report for {identifier}.')
      return result
    result['line_cov'], result['branch_cov'] = _extract_coverage(report_path, item)

    cmd_mutate = mvn_prefix + ['clean', 'test-compile', 'org.pitest:pitest-maven:mutationCoverage',
                               '-Drat.skip=true', '-Dsurefire.failIfNoSpecifiedTests=false',
                               '-Dcheckstyle.skip', '-Dmaven.gitcommitid.skip',
                               f'-DtargetClasses={item["package"]}.{item["class_name"]}',
                               f'-DtargetTests={item["package"]}.{classname}']
    try:
      completed = subprocess.run(cmd_mutate, cwd=repo_root, capture_output=True, check=True,
                                 encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
    except subprocess.CalledProcessError as e:
      logger.warning(f'Failed to run mutation test on {identifier}:\n{e.stdout}')
      return result
    except subprocess.TimeoutExpired:
      logger.warning(f'Mutation test of {identifier} timed out.')
      return result
    result['mut_score'] = _extract_mutation_score(completed.stdout)
    return result


_REPO_LOCKS: dict[Path, Lock] = defaultdict(Lock)


@contextmanager
def _git_worktree(repo_root: Path) -> Iterator[Path]:
  with tempfile.TemporaryDirectory() as tmpdir:
    worktree_path = Path(tmpdir)
    with _REPO_LOCKS[repo_root]:
      cmd_add = ['git', 'worktree', 'add', '--detach', '-f', str(worktree_path), 'HEAD']
      try:
        subprocess.run(cmd_add, cwd=repo_root, capture_output=True, check=True, encoding='utf-8')
      except subprocess.CalledProcessError as e:
        logger.error(f'Failed to create worktree for {repo_root}:\n{e.stderr}')
        raise

    try:
      cmd_diff = ['git', 'diff', 'HEAD', '--binary']
      if diff := subprocess.check_output(cmd_diff, cwd=repo_root):
        cmd_apply = ['git', 'apply', '--whitespace=nowarn', '-']
        subprocess.run(cmd_apply, cwd=worktree_path, capture_output=True, check=True, input=diff)
    except subprocess.CalledProcessError as e:
      logger.warning(f'Failed to sync dirty state to {worktree_path} for {repo_root}:\n{e.stderr}')

    try:
      yield worktree_path
    finally:
      with _REPO_LOCKS[repo_root]:
        cmd_remove = ['git', 'worktree', 'remove', '-f', str(worktree_path)]
        try:
          subprocess.run(cmd_remove, cwd=repo_root, capture_output=True, check=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
          logger.warning(f'Failed to remove worktree for {repo_root}:\n{e.stderr}')
          cmd_prune = ['git', 'worktree', 'prune']
          subprocess.run(cmd_prune, cwd=repo_root, capture_output=True, encoding='utf-8')


def _get_submodule(relative_path: str) -> str:
  if matched := re.match(r'^(.+)/src', relative_path):
    return matched.group(1)
  return ''


def _extract_coverage(report_path: str, item: dict[str, Any]) -> tuple[float | None, float | None]:
  try:
    tree = ET.parse(report_path)
  except ET.ParseError as e:
    logger.warning(f'Failed to parse coverage report:\n{e}')
    return None, None
  root = tree.getroot()
  try:
    package = next(p for p in root.findall(".//package")
                   if p.get('name') == item['package'].replace('.', '/'))
    clazz = next(c for c in package.findall('class')
                 if (name := c.get('name')) and name.endswith('/' + item['class_name']))
    line_cnt, branch_cnt = 0, 0
    line_cov, branch_cov = .0, .0
    for method in clazz.findall('method'):
      if method.get('name') == item['method_name']:
        # Element.__bool__ stands for whether it has children
        if (line_ctr := method.find('counter[@type="LINE"]')) is not None:
          line_cnt += 1
          line_cov += _calculate_coverage(line_ctr)
        if (branch_ctr := method.find('counter[@type="BRANCH"]')) is not None:
          branch_cnt += 1
          branch_cov += _calculate_coverage(branch_ctr)
    return (line_cov / line_cnt if line_cnt else .0,
            branch_cov / branch_cnt if branch_cnt else .0)
  except StopIteration:
    logger.verbose(f'Failed to find coverage info for {item["class_name"]}::{item["method_name"]}.')
    return None, None


def _calculate_coverage(counter: ET.Element) -> float:
  missed = int(counter.get('missed', 0))
  covered = int(counter.get('covered', 0))
  total = missed + covered
  return covered / total if total else .0


def _extract_mutation_score(stdout: str) -> float | None:
  if matches := re.search(r'Generated \d+ mutations Killed \d+ \((\d{1,3})%\)', stdout):
    return int(matches.group(1)) / 100
  return None
