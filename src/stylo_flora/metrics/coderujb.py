"""
Modified from CoderUJB repository (https://github.com/ZZR0/CoderUJB).
"""

import os
import subprocess
import tempfile
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm import tqdm

from stylo_flora import setting_dict
from stylo_flora.logger import logger


def pass_at_1_ujb(patches: Seq[str], items: Seq[dict[str, Any]], lang: str) -> float:
  if lang != 'java':
    raise ValueError(f'Unsupported language: {lang}')
  with ThreadPoolExecutor(max_workers=setting_dict['metrics']['max_workers']) as executor:
    return sum(tqdm(executor.map(_validate_all_patches, patches, items),
                    desc='Calculating Count@1', total=len(patches), leave=False)) / len(patches)


def _validate_all_patches(patch: str, item: dict[str, Any]) -> bool:
  with tempfile.TemporaryDirectory() as tmpdir:
    cmd_checkout = ['defects4j', 'checkout', '-p', item['project'],
                    '-v', f'{item["bug_id"]}f', '-w', tmpdir]
    try:
      subprocess.run(cmd_checkout, env=_get_d4j_env(), capture_output=True,
                     check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
      logger.warning(f'Failed to checkout project:\n{e.stderr}')
      return False

    source_lines = item['source'].split('\n')
    patch_lines = patch.split('\n')
    source = '\n'.join(source_lines[:item['start']] + patch_lines + source_lines[item['end'] + 1:])
    with open(os.path.join(tmpdir, item['location']), 'w') as f:
      f.write(source)

    cmd_test = ['defects4j', 'test', '-w', tmpdir]
    try:
      completed = subprocess.run(cmd_test, env=_get_d4j_env(), capture_output=True, check=True,
                                 encoding='utf-8', timeout=setting_dict['agent']['timeout'])
      return 'Failing tests: 0\n' in completed.stdout
    except subprocess.CalledProcessError as e:
      logger.verbose(f'Failed to test project:\n{e.stderr}')
    except subprocess.TimeoutExpired:
      logger.verbose('Test timed out.')
    return False


def _get_d4j_env() -> dict[str, str]:
  d4j_java_home = os.getenv('D4J_JAVA_HOME')
  if not d4j_java_home:
    logger.warning('D4J_JAVA_HOME is not set, using default JAVA_HOME.')
    return dict(os.environ)
  return os.environ | {
      'JAVA_HOME': d4j_java_home,
      'PATH': os.path.join(d4j_java_home, 'bin') + os.pathsep + os.getenv('PATH', ''),
  }
