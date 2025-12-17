"""
Modified from CoderUJB repository (https://github.com/ZZR0/CoderUJB).
"""

import os
import signal
import subprocess
import tempfile
from collections import Counter
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from javalang import parser, tokenizer
from tqdm import tqdm

from .. import setting_dict
from ..logger import logger
from .correctness import CorrectnessResult
from .utils import Correctness


def pass_at_1_ujb(
    patches: Seq[str],
    items: Seq[dict[str, Any]],
    lang: str
) -> CorrectnessResult:
  if lang != 'java':
    raise ValueError(f'Unsupported language: {lang}')
  with ThreadPoolExecutor(max_workers=setting_dict['metrics']['max_workers']) as executor:
    counter = Counter(tqdm(executor.map(_validate_all_patches, patches, items),
                           desc='Calculating Pass@1', total=len(patches), leave=False))
  total = sum(counter.values())
  return CorrectnessResult(
      comp_rate=(total - counter[Correctness.FAIL_COMP]) / total,
      pass_rate=counter[Correctness.PASS] / total,
  )


def _validate_all_patches(patch: str, item: dict[str, Any]) -> Correctness:
  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
    cmd_checkout = ['defects4j', 'checkout', '-p', item['project'],
                    '-v', f'{item["bug_id"]}f', '-w', tmpdir]
    try:
      subprocess.run(cmd_checkout, env=_get_d4j_env(), capture_output=True,
                     check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
      logger.warning(f'Failed to checkout {item["project"]}:\n{e.stderr}')
      return Correctness.FAIL_COMP

    source_lines = item['source'].split('\n')
    patch_lines = patch.split('\n')
    source = '\n'.join(source_lines[:item['start']] + patch_lines + source_lines[item['end'] + 1:])
    with open(os.path.join(tmpdir, item['location']), 'w') as f:
      f.write(source)

    try:
      tokens = tokenizer.tokenize(source)
      parser.Parser(tokens).parse()
    except Exception as e:
      logger.verbose(f'{e.__class__.__name__} occurred while compiling on {item["project"]}.')
      return Correctness.FAIL_COMP

    cmd_test = ['defects4j', 'test', '-w', tmpdir]
    try:
      with subprocess.Popen(
          cmd_test, env=_get_d4j_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
          encoding='utf-8', start_new_session=True,
      ) as process:
        try:
          stdout, stderr = process.communicate(timeout=setting_dict['agent']['timeout'])
          if process.returncode != 0 or 'Failing tests: 0\n' not in stdout:
            logger.verbose(f'Failed to test on {item["project"]} {item["bug_id"]}:\n{stderr}')
            return Correctness.FAIL_EXEC
        except subprocess.TimeoutExpired:
          logger.verbose(f'Test timed out on {item["project"]} {item["bug_id"]}.')
          try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
          except ProcessLookupError:
            pass
          return Correctness.FAIL_EXEC
    except Exception as e:
      logger.verbose(f'Failed to test on {item["project"]} {item["bug_id"]}:\n{str(e)}')
      return Correctness.FAIL_COMP
    return Correctness.PASS


def _get_d4j_env() -> dict[str, str]:
  d4j_java_home = os.getenv('D4J_JAVA_HOME')
  if not d4j_java_home:
    logger.warning('D4J_JAVA_HOME is not set, using default JAVA_HOME.')
    return dict(os.environ)
  return os.environ | {
      'JAVA_HOME': d4j_java_home,
      'PATH': os.path.join(d4j_java_home, 'bin') + os.pathsep + os.getenv('PATH', ''),
  }
