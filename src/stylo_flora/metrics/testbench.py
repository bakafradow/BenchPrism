"""
Modified from TestBench repository (https://github.com/iSEngLab/TestBench).
"""

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence as Seq
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from tqdm import tqdm

from .. import setting_dict
from ..logger import logger
from .utils import Correctness, extract_classname_java

PROJ_PATH = Path(setting_dict['datasets']['testbench_root']) / 'java_project'


class CoverageResultTB(NamedTuple):
  comp_rate: float
  pass_rate: float
  line_cov: float
  branch_cov: float
  mut_score: float


def calc_coverage_tb(tests: Seq[str], items: Seq[dict[str, Any]], lang: str) -> CoverageResultTB:
  if lang != 'java':
    raise ValueError(f'Unsupported language: {lang}')
  # concurrecy may introduce error
  dicts = list(tqdm(map(_worker, tests, items),
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


PATTERN_MAVEN_STAT = re.compile(r'Tests run: (\d+), Failures: (\d+), Errors: (\d+)')


def _worker(test: str, item: dict[str, Any]) -> dict[str, Any]:
  result: dict[str, Any] = {'correctness': Correctness.FAIL_COMP}
  identifier = f'{item["class_name"]}::{item["method_name"]}'
  proj_root = PROJ_PATH / item['project_name']
  test_dir = PROJ_PATH / item['relative_path'].replace('main', 'test').rsplit('/', 1)[0]
  test_dir.mkdir(parents=True, exist_ok=True)
  classname = extract_classname_java(test)
  if not classname:
    logger.verbose(f'Failed to extract class name for {identifier} test.')
    return result

  _clean_repo(proj_root)
  with open(test_dir / f'{classname}.java', 'w') as f:
    f.write(test)
  mvn_prefix = ['mvn', '-B', '-Dmaven.compiler.showWarnings=false']
  if matched := re.match(r'^[\w-]+/(.+)/src', item['relative_path']):
    submodule = matched.group(1)
    mvn_prefix += ['-pl', submodule, '-am']  # avoid building unnecessary modules
  else:
    submodule = ''
  cmd_compile = mvn_prefix + ['-q', 'clean', 'test-compile', '-Drat.skip=true',
                              '-Dsurefire.failIfNoSpecifiedTests=false', '-Dcheckstyle.skip']
  try:
    subprocess.run(cmd_compile, cwd=proj_root, capture_output=True, check=True,
                   encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
  except subprocess.CalledProcessError as e:
    logger.verbose(f'Failed to compile {identifier}\n{e.stdout}')
    return result
  except subprocess.TimeoutExpired:
    logger.verbose(f'Compilation of {identifier} timed out.')
    return result
  result['correctness'] = Correctness.FAIL_EXEC

  cmd_test = mvn_prefix + ['clean', 'test', '-DskipPitest=True',
                           '-Dsurefire.failIfNoSpecifiedTests=false',
                           '-Dcheckstyle.skip', f'-Dtest={classname}']
  try:
    completed = subprocess.run(cmd_test, cwd=proj_root, capture_output=True, encoding='utf-8',
                               timeout=setting_dict['metrics']['timeout'])
    matched = PATTERN_MAVEN_STAT.search(completed.stdout)
    if not matched:
      logger.warning(f'Failed to parse test result for {identifier}.')
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
  report_path = os.path.join(proj_root, submodule, 'target/site/jacoco/jacoco.xml')
  if not os.path.exists(report_path):
    logger.warning(f'Failed to find coverage report for {identifier}.')
    return result
  result['line_cov'], result['branch_cov'] = _extract_coverage(report_path, item)

  cmd_mutate = mvn_prefix + ['clean', 'test-compile', 'org.pitest:pitest-maven:mutationCoverage',
                             '-Drat.skip=true', '-Dsurefire.failIfNoSpecifiedTests=false',
                             '-Dcheckstyle.skip',
                             f'-DtargetClasses={item["package"]}.{item["class_name"]}',
                             f'-DtargetTests={item["package"]}.{classname}']
  try:
    completed = subprocess.run(cmd_mutate, cwd=proj_root, capture_output=True, check=True,
                               encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
  except subprocess.CalledProcessError as e:
    logger.warning(f'Failed to run mutation test on {identifier}:\n{e.stdout}')
    return result
  except subprocess.TimeoutExpired:
    logger.warning(f'Mutation test of {identifier} timed out.')
    return result
  result['mut_score'] = _extract_mutation_score(completed.stdout)
  return result


SKIP_FILES = {'pom.xml', 'DynamicRouteService.java'}


def _clean_repo(proj_root: os.PathLike) -> None:
  cmd_status = ['git', 'status', '--porcelain']
  try:
    completed = subprocess.run(cmd_status, cwd=proj_root, capture_output=True,
                               check=True, encoding='utf-8')
  except subprocess.CalledProcessError as e:
    logger.warning(f'Failed to check git status:\n{e.stderr}')
    return
  for status, path in [line.split() for line in completed.stdout.splitlines()]:
    if any(s in path for s in SKIP_FILES):
      continue
    if status == 'M':
      cmd_checkout = ['git', 'checkout', '-f', path]
      try:
        subprocess.run(cmd_checkout, cwd=proj_root, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True, encoding='utf-8')
      except subprocess.CalledProcessError as e:
        logger.warning(f'Failed to checkout {path}:\n{e.stderr}')
    elif status == '??':
      cmd_clean = ['git', 'clean', '-f', path]
      try:
        subprocess.run(cmd_clean, cwd=proj_root, capture_output=True, check=True, encoding='utf-8')
      except subprocess.CalledProcessError as e:
        logger.warning(f'Failed to clean {path}:\n{e.stderr}')


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
