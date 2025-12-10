import subprocess
import os
import sys
import tempfile
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import IOTestCase, setting_dict
from ..logger import logger
from .utils import extract_classname_java


def calc_coverage_java(code: str, tc_list: Seq[IOTestCase]) -> dict:
  with tempfile.TemporaryDirectory() as temp_dir:
    ORIGINAL_DIR = os.path.join(temp_dir, 'original')
    INSTRUMENT_DIR = os.path.join(temp_dir, 'instrumented')
    EXEC_PATH = os.path.join(temp_dir, 'test.exec')
    REPORT_PATH = os.path.join(temp_dir, 'coverage.csv')

    classname = extract_classname_java(code)
    if not classname:
      logger.warning('Failed to extract class name.')
      return {}
    src_path = os.path.join(temp_dir, f'{classname}.java')
    with open(src_path, 'w') as f:
      f.write(code)
    cmd_compile = ['javac', '-d', ORIGINAL_DIR, src_path]
    completed = subprocess.run(cmd_compile, cwd=temp_dir, capture_output=True, encoding='utf-8',
                               timeout=setting_dict['metrics']['timeout'])
    if completed.returncode != 0:
      logger.warning(f'Failed to compile {classname}.')
      return {}

    cmd_instrument = ['java', 'org.jacoco.cli.internal.Main', 'instrument',
                      ORIGINAL_DIR, '--dest', INSTRUMENT_DIR]
    try:
      completed = subprocess.run(cmd_instrument, cwd=temp_dir, capture_output=True,
                                 check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
      logger.warning(f'Failed to instrument {classname} with jacocoagent:\n{e.stderr}')
      return {}

    cmd_exec = ['java', f'-Djacoco-agent.destfile={EXEC_PATH}',
                '-Djacoco-agent.append=true', classname]
    def worker(testcase: IOTestCase) -> bool:
      try:
        completed = subprocess.run(cmd_exec, cwd=INSTRUMENT_DIR, input=testcase.input,
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
    pass_rate = num_pass / len(tc_list)

    cmd_report = ['java', 'org.jacoco.cli.internal.Main', 'report', EXEC_PATH,
                  '--classfiles', ORIGINAL_DIR, '--sourcefiles', '.', '--csv', REPORT_PATH]
    try:
      completed = subprocess.run(cmd_report, cwd=temp_dir, capture_output=True,
                                 check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
      logger.warning(f'Failed to generate coverage report for {classname}:\n{e.stderr}')
      return {'pass_rate': pass_rate}
    with open(REPORT_PATH, 'r') as f:
      return pd.read_csv(f).to_dict(orient='records')[0] | {'pass_rate': pass_rate}


def calc_coverage(code_list: Seq[str], tc_lists: Seq[Seq[IOTestCase]], lang: str) -> dict:
  coverage_func = getattr(sys.modules[__name__], f'calc_coverage_{lang}', None)
  if not coverage_func:
    raise TypeError(f'Unsupported language: {lang}')
  coverages = list(tqdm((coverage_func(code, tc_list)
                         for code, tc_list in zip(code_list, tc_lists)),
                        desc='Calculating coverage', total=len(code_list)))
  return {
      'pass_rate': np.mean([cov['pass_rate'] for cov in coverages]),
      'line_cov_rate': np.mean([cov['LINE_COVERED'] / (cov['LINE_COVERED'] + cov['LINE_MISSED']) if (cov['LINE_COVERED'] + cov['LINE_MISSED']) > 0 else 1
                                for cov in coverages if cov]),
      'branch_cov_rate': np.mean([cov['BRANCH_COVERED'] / (cov['BRANCH_COVERED'] + cov['BRANCH_MISSED']) if (cov['BRANCH_COVERED'] + cov['BRANCH_MISSED']) > 0 else 1
                                  for cov in coverages if cov]),
  }
