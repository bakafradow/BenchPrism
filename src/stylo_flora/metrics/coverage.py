import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import IOTestCase, setting_dict
from ..logger import logger
from .utils import Correctness, extract_classname_java


class CoverageResult(NamedTuple):
  comp_rate: float
  pass_rate: float
  line_cov: float
  branch_cov: float


def calc_coverage(code_list: Seq[str], tc_lists: Seq[Seq[IOTestCase]], lang: str) -> CoverageResult:
  coverage_func = getattr(sys.modules[__name__], f'calc_coverage_{lang}', None)
  if not coverage_func:
    raise TypeError(f'Unsupported language: {lang}')
  dicts = list(tqdm((coverage_func(code, tc_list) for code, tc_list in zip(code_list, tc_lists)),
                    desc='Calculating coverage', total=len(code_list), leave=False))
  counter = Counter([d['correctness'] for d in dicts])
  total = sum(counter.values())
  return CoverageResult(
      comp_rate=(total - counter[Correctness.FAIL_COMP]) / total,
      pass_rate=np.mean([d['pass_rate'] for d in dicts if d.get('pass_rate')]),
      line_cov=np.mean([d['line_cov'] for d in dicts if d.get('line_cov')]),
      branch_cov=np.mean([d['branch_cov'] for d in dicts if d.get('branch_cov')]),
  )


def calc_coverage_java(code: str, tc_list: Seq[IOTestCase]) -> dict[str, Any]:
  result: dict[str, Any] = {'correctness': Correctness.FAIL_COMP}
  with tempfile.TemporaryDirectory() as temp_dir:
    original_dir = os.path.join(temp_dir, 'original')
    instrument_dir = os.path.join(temp_dir, 'instrumented')
    exec_path = os.path.join(temp_dir, 'test.exec')
    report_path = os.path.join(temp_dir, 'coverage.csv')

    classname = extract_classname_java(code)
    if not classname:
      logger.verbose('Failed to extract class name.')
      return result
    src_path = os.path.join(temp_dir, f'{classname}.java')
    with open(src_path, 'w') as f:
      f.write(code)
    cmd_compile = ['javac', '-d', original_dir, src_path]
    try:
      subprocess.run(cmd_compile, cwd=temp_dir, capture_output=True, check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Failed to compile {classname}:\n{e.stderr}')
      return result

    cmd_instrument = ['java', 'org.jacoco.cli.internal.Main', 'instrument',
                      original_dir, '--dest', instrument_dir]
    try:
      subprocess.run(cmd_instrument, cwd=temp_dir, capture_output=True, check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Failed to instrument {classname} with jacocoagent:\n{e.stderr}')
      return result
    result['correctness'] = Correctness.FAIL_EXEC

    cmd_exec = ['java', f'-Djacoco-agent.destfile={exec_path}',
                '-Djacoco-agent.append=true', classname]

    def worker(testcase: IOTestCase) -> bool:
      try:
        completed = subprocess.run(cmd_exec, cwd=instrument_dir, input=testcase.input,
                                   capture_output=True, encoding='utf-8')
        if completed.returncode != 0:
          logger.verbose(f'Failed to execute {classname} with jacocoagent:\n{completed.stderr}')
          return False
        return completed.stdout.strip() in (output.strip() for output in testcase.outputs)
      except Exception as e:
        logger.warning(f'{e.__class__.__name__} occurred while executing instrumented {classname}.')
        return False

    with ThreadPoolExecutor(max_workers=setting_dict['metrics']['max_workers']) as executor:
      num_pass = sum(tqdm(executor.map(worker, tc_list), total=len(tc_list), leave=False))
    result['pass_rate'] = num_pass / len(tc_list)

    cmd_report = ['java', 'org.jacoco.cli.internal.Main', 'report', exec_path,
                  '--classfiles', original_dir, '--sourcefiles', '.', '--csv', report_path]
    try:
      subprocess.run(cmd_report, cwd=temp_dir, capture_output=True, check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
      logger.warning(f'Failed to generate coverage report for {classname}:\n{e.stderr}')
      return result
    result['line_cov'], result['branch_cov'] = _extract_coverage(report_path)
    return result


def _extract_coverage(report_path: str) -> tuple[float | None, float | None]:
  with open(report_path, 'r') as f:
    report_dict = pd.read_csv(f).to_dict(orient='records')[0]
  line_cov, branch_cov = None, None
  if line_total := report_dict['LINE_COVERED'] + report_dict['LINE_MISSED']:
    line_cov = report_dict['LINE_COVERED'] / line_total
  if branch_total := report_dict['BRANCH_COVERED'] + report_dict['BRANCH_MISSED']:
    branch_cov = report_dict['BRANCH_COVERED'] / branch_total
  return line_cov, branch_cov
