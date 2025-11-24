import subprocess
import sys
import tempfile
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import IOTestCase, setting_dict
from ..logger import logger
from .utils import extract_classname_java


def calc_coverage_java(code: str, tc_list: Seq[IOTestCase]) -> dict:
  with tempfile.TemporaryDirectory() as temp_dir:
    classname = extract_classname_java(code)
    if not classname:
      logger.warning('Failed to extract class name.')
      return {}
    with open(f'{temp_dir}/{classname}.java', 'w') as f:
      f.write(code)
    compile_cmd = f'javac {temp_dir}/{classname}.java'.split()
    returned = subprocess.run(compile_cmd, cwd=temp_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
    if returned.returncode != 0:
      logger.warning('Failed to compile snippet.')
      return {}

    jar_dir = Path('data').absolute()
    exec_name = 'test.exec'
    csv_name = 'coverage.csv'
    def worker(testcase: IOTestCase) -> bool:
      exec_cmd = f'java -javaagent:{jar_dir}/jacocoagent.jar=destfile={exec_name},append=true {classname}'.split()
      returned = subprocess.run(exec_cmd, cwd=temp_dir, input=testcase.input,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
      if returned.returncode != 0:
        logger.warning(f'Failed to execute snippet with jacocoagent:\n{returned.stdout}')
        return False
      return returned.stdout.strip() in (output.strip() for output in testcase.outputs)

    with ThreadPoolExecutor(max_workers=setting_dict['metrics']['max_workers']) as executor:
      num_pass = sum(tqdm(executor.map(worker, tc_list), total=len(tc_list), leave=False))
    pass_rate = num_pass / len(tc_list)

    report_cmd = f'java -jar {jar_dir}/jacococli.jar report {exec_name} --classfiles . --sourcefiles . --csv {csv_name}'.split()
    returned = subprocess.run(report_cmd, cwd=temp_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding='utf-8', timeout=setting_dict['metrics']['timeout'])
    if returned.returncode != 0:
      logger.warning('Failed to generate coverage report.')
      return {'pass_rate': pass_rate}
    with open(f'{temp_dir}/{csv_name}', 'r') as f:
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
