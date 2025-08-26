import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .. import Snippet
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
    num_pass = 0
    for testcase in tqdm(snippet.args['io_testcases'], desc='Calculating coverage',
                         total=len(snippet.args['io_testcases']), leave=False):
      exec_cmd = f'java -javaagent:{jar_dir}/jacocoagent.jar=destfile={exec_name},append=true {classname}'.split()
      returned = subprocess.run(exec_cmd, cwd=temp_dir, input=testcase.input,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                encoding='utf-8', timeout=config['timeout'])
      if returned.returncode != 0:
        logger.warning(f'Failed to execute snippet {snippet.id} with jacocoagent.')
        continue
      num_pass += returned.stdout.strip() in (output.strip() for output in testcase.outputs)
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
  coverages = [coverage_func(snippet) for snippet in snippets]
  return {
      'pass_rate': np.mean([cov['pass_rate'] for cov in coverages]),
      'line_cov_rate': np.mean([cov['LINE_COVERED'] / (cov['LINE_COVERED'] + cov['LINE_MISSED'])
                                for cov in coverages]),
      'branch_cov_rate': np.mean([cov['BRANCH_COVERED'] / (cov['BRANCH_COVERED'] + cov['BRANCH_MISSED'])
                                  for cov in coverages]),
  }
