import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .. import IOTestCase, Snippet
from ..logger import logger
from .utils import extract_classname_java

with open('settings.yml') as f:
  config = yaml.safe_load(f)['metrics']


def calc_coverage_java(snippet: Snippet) -> dict:
  with tempfile.TemporaryDirectory() as temp_dir:
    classname = extract_classname_java(snippet)
    if not classname:
      logger.warning(f'Failed to extract class name from snippet {snippet.id}.')
      return {}
    with open(f'{temp_dir}/{classname}.java', 'w') as f:
      f.write(snippet.code)
    compile_cmd = f'javac {temp_dir}/{classname}.java'.split()
    returned = subprocess.run(compile_cmd, cwd=temp_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding='utf-8', timeout=config['timeout'])
    if returned.returncode != 0:
      logger.warning(f'Failed to compile snippet {snippet.id}.')
      return {}

    jar_dir = Path('data').absolute()
    exec_name = 'test.exec'
    csv_name = 'coverage.csv'
    def worker(testcase: IOTestCase) -> bool:
      exec_cmd = f'java -javaagent:{jar_dir}/jacocoagent.jar=destfile={exec_name},append=true {classname}'.split()
      returned = subprocess.run(exec_cmd, cwd=temp_dir, input=testcase.input,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding='utf-8', timeout=config['timeout'])
      if returned.returncode != 0:
        logger.warning(f'Failed to execute snippet {snippet.id} with jacocoagent:\n{returned.stdout}')
        return False
      return returned.stdout.strip() in (output.strip() for output in testcase.outputs)

    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
      num_pass = sum(tqdm(executor.map(worker, snippet.args['io_testcases']),
                          total=len(snippet.args['io_testcases']), leave=False))
    pass_rate = num_pass / len(snippet.args['io_testcases'])

    report_cmd = f'java -jar {jar_dir}/jacococli.jar report {exec_name} --classfiles . --sourcefiles . --csv {csv_name}'.split()
    returned = subprocess.run(report_cmd, cwd=temp_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              encoding='utf-8', timeout=config['timeout'])
    if returned.returncode != 0:
      logger.warning(f'Failed to generate coverage report for snippet {snippet.id}.')
      return {'pass_rate': pass_rate}
    with open(f'{temp_dir}/{csv_name}', 'r') as f:
      return pd.read_csv(f).to_dict(orient='records')[0] | {'pass_rate': pass_rate}


def calc_coverage(snippets: Sequence[Snippet], lang: str) -> dict:
  coverage_func = globals().get(f'calc_coverage_{lang}')
  if not coverage_func:
    raise TypeError(f'Unsupported language: {lang}')
  coverages = list(tqdm((coverage_func(snippet) for snippet in snippets),
                        desc='Calculating coverage', total=len(snippets)))
  return {
      'pass_rate': np.mean([cov['pass_rate'] for cov in coverages]),
      'line_cov_rate': np.mean([cov['LINE_COVERED'] / (cov['LINE_COVERED'] + cov['LINE_MISSED']) if (cov['LINE_COVERED'] + cov['LINE_MISSED']) > 0 else 1
                                for cov in coverages if cov]),
      'branch_cov_rate': np.mean([cov['BRANCH_COVERED'] / (cov['BRANCH_COVERED'] + cov['BRANCH_MISSED']) if (cov['BRANCH_COVERED'] + cov['BRANCH_MISSED']) > 0 else 1
                                  for cov in coverages if cov]),
  }
