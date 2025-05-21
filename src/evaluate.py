import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Sequence

import jsonlines
import pandas as pd
import yaml
from tqdm import tqdm

from . import Snippet
from .utils import extract_field_from, logger

with open('settings.yml') as f:
  config = yaml.safe_load(f)['evaluator']
TARGET_DIR = Path(config['target_dir'])
RESULT_DIR = Path(config['result_dir'])
os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


class CompilationError(Exception):
  def __init__(self, id: int, message: str, stderr: str = ''):
    self.stderr = stderr
    super().__init__(f'{id}: {message}')


def _compile(snippet: Snippet, lang: str) -> str:
  """
  Compiles the code snippet.
  :param code: the code snippet to be compiled
  :param lang: the language of the code snippet
  :return: the path of the compiled executable or the code itself for interpreted languages
  """
  match lang:
    case 'java':
      matched = re.search(r'public\s+(?:final\s+)?class\s+(\w+)', snippet.code)
      if not matched:
        matched = re.search(r'(?:final\s+)?class\s+(Main|Solution)', snippet.code)
      if not matched:
        raise CompilationError(snippet.id, 'Failed to extract class name from Java code.', f'Generated code:\n{snippet.code}')
      classname = matched.group(1)
      with tempfile.TemporaryDirectory() as tmpdir, open(f'{tmpdir}/{classname}.java', 'w') as f:
        f.write(snippet.code)
        f.flush()
        classdir = tempfile.mkdtemp(dir=TARGET_DIR)
        try:
          returned = subprocess.run(['javac', '-d', classdir, f.name], stderr=subprocess.PIPE, encoding='utf-8', timeout=config['timeout'])
        except subprocess.TimeoutExpired:
          raise CompilationError(snippet.id, f'Compilation of {f.name} timed out.')
        if returned.returncode != 0:
          raise CompilationError(snippet.id, f'Failed to compile {f.name}.', returned.stderr)
      return os.path.join(os.path.basename(classdir), classname)
    case 'cpp':
      with tempfile.NamedTemporaryFile(suffix='.cpp') as f:
        f.write(snippet.code.encode())
        f.flush()
        executable = re.sub(r'\.cpp$', '', f.name)
        try:
          returned = subprocess.run(['g++', f.name, '-o', executable], stderr=subprocess.PIPE, encoding='utf-8', timeout=config['timeout'])
        except subprocess.TimeoutExpired:
          raise CompilationError(snippet.id, f'Compilation of {f.name} timed out.')
        if returned.returncode != 0:
          raise CompilationError(snippet.id, f'Failed to compile {f.name}.', returned.stderr)
      return executable
    case 'python':
      return snippet.code
    case _:
      raise TypeError(f'Unsupported language: {lang}.')


def run_with_assertion(snippet: Snippet, test: Sequence[dict], lang: str) -> bool:
  """
  Runs the code snippet with the test which asserts the correctness of the code.
  :param code: the code snippet to be tested WITHOUT main function
  :param test: the code snippet that contains the main function of the test
  :param lang: the language of the code snippet
  :return: whether the code snippet passes the test
  """
  try:
    executable = _compile(snippet._replace(code=snippet.code + test), lang)
  except CompilationError as err:
    logger.warning(err)
    logger.verbose(err.stderr)
    return False
  match lang:
    case 'java':
      parent, classname = os.path.split(executable)
      args = ['java', '-classpath', f'{TARGET_DIR / parent}']
      executable = classname
    case 'cpp':
      args = []
    case 'python':
      args = ['python', '-c']
    case _:
      raise TypeError(f'Unsupported language: {lang}.')
  try:
    returned = subprocess.run(args + [executable], stderr=subprocess.PIPE, encoding='utf-8', timeout=config['timeout'])
  except KeyboardInterrupt:
    logger.warning('Keyboard interrupt.')
    raise
  except subprocess.TimeoutExpired:
    logger.verbose(f'Time out on {snippet.id}.')
    return False
  except Exception as e:
    logger.verbose(f'Error on {snippet.id}: {e}')
    return False
  if returned.returncode != 0:
    logger.verbose(f'Failed assertion on {snippet.id}:\n{returned.stderr}')
    return False
  return True


def run_with_io(snippet: Snippet, test: Sequence[dict], lang: str) -> bool:
  """
  Runs the code snippet with the test which checks the input-output behavior of the code.
  :param code: the code snippet to be tested with main function
  :param test: the input-output pair of the test
  :param lang: the language of the code snippet
  :return: whether the code snippet passes the test
  """
  try:
    executable = _compile(snippet, lang)
  except CompilationError as err:
    logger.warning(err)
    logger.verbose(err.stderr)
    return False
  match lang:
    case 'java':
      parent, classname = os.path.split(executable)
      args = ['java', '-classpath', f'{TARGET_DIR / parent}']
      executable = classname
    case 'cpp':
      args = []
    case 'python':
      args = ['python', '-c']
    case _:
      raise TypeError(f'Unsupported language: {lang}.')
  for pair in tqdm(test, desc='Running tests', total=len(test), leave=False):
    try:
      returned = subprocess.run(args + [executable], input=pair['input'], text=True, capture_output=True, encoding='utf-8', timeout=config['timeout'])
    except KeyboardInterrupt:
      logger.warning('Keyboard interrupt.')
      raise
    except subprocess.TimeoutExpired:
      logger.verbose(f'Time out on {snippet.id} with input {pair["input"].strip()}.')
      return False
    except Exception as e:
      logger.verbose(f'Error on {snippet.id} with input {pair["input"].strip()}: {e}')
      return False
    if returned.returncode != 0:
      logger.verbose(f'{snippet.id} returned {returned.returncode} on input {pair["input"].strip()}\nStandard Error:\n{returned.stderr}')
      return False
    if returned.stdout.strip() != pair['output'][0].strip():
      logger.verbose(f'{snippet.id} failed on input {pair["input"].strip()}\nExpected:\n{pair["output"][0]}\nActual:\n{returned.stdout}')
      return False
  return True


def calculate_correctness(dataset: str, snippets: Sequence[Snippet], tests: Sequence[Sequence[dict]], lang: str) -> float:
  """
  Checks the correctness of the translated code with the tests.
  :param dataset: the dataset name
  :param snippets: the translated code snippets
  :param tests: the tests
  :param lang: the language of the code snippets
  """
  def worker(snippet: Snippet, test: Sequence[dict]) -> bool:
    if not snippet:
      return False
    match dataset:
      case 'HumanEvalX':
        return run_with_assertion(snippet, test, lang)
      case 'xCodeEval' | 'CodeNet':
        return run_with_io(snippet, test, lang)
      case _:
        raise TypeError(f'Unsupported dataset: {dataset}.')
  max_workers = min(max(1, config['max_workers']), os.cpu_count())
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(worker, snippets, tests), desc='Calculating correctness', total=len(snippets), leave=False))
  return sum(results) / len(snippets)


def load_tests(dataset: str, src_lang: str, dst_lang: str, ids: Sequence[str], *, translated: bool = False) -> Sequence[Sequence[dict]]:
  match dataset:
    case 'HumanEvalX':
      tests = extract_field_from(dataset, dst_lang if translated else src_lang, 'test')
    case 'xCodeEval':
      with open('data/xCodeEval/unittest_db.json', 'r') as f:
        unittests = json.load(f)
      tests = [unittests[uid] for uid in ids]
      for test in tests:
        for pair in test:
          pair['input'] = pair['input'].replace('\r\n', '\n')
          pair['output'] = [line.replace('\r\n', '\n') for line in pair['output']]
    case 'CodeNet':
      with jsonlines.open('data/CodeNet/codenet_test.jsonl', 'r') as reader:
        tests_dict = {obj['id']: obj['test'] for obj in reader}
      tests = [[{'input': pair[0], 'output': [pair[1]]} for pair in tests_dict[id_]] for id_ in ids]
    case 'CodeXGLUE':
      raise NotImplementedError('CodeXGLUE dataset does not provide tests.')
    case _:
      raise TypeError(f'Unknown dataset: {dataset}.')
  return tests


def evaluate(dataset: str, snippets: Sequence[Snippet], variants: Sequence[Snippet], translated_snippets: Sequence[Snippet], translated_variants: Sequence[Snippet], src_lang: str, dst_lang: str) -> None:
  """
  Evaluates the space spanned by the translated code relative to the original source code
  :param dataset: dataset name
  :param variants: the original transformed code snippets
  :param translated_snippets: the translated original code snippets
  :param translated_variants: the translated transformed code snippets
  :param src_lang: the source language of the code snippets
  :param dst_lang: the target language of the code snippets
  """
  logger.info(f'Evaluating {len(snippets)} snippets on {dataset}...')
  if not len(variants) == len(translated_snippets) == len(translated_variants):
    raise ValueError('The number of snippets and variants should equal.')
  ids = tuple(snippet.id for snippet in snippets)
  src_tests = load_tests(dataset, src_lang, dst_lang, ids)[:len(variants)]
  dst_tests = load_tests(dataset, src_lang, dst_lang, ids, translated=True)[:len(variants)]
  logger.info('Testing originals.')
  original_correctness = calculate_correctness(dataset, snippets, src_tests, src_lang)
  logger.info('Testing variants.')
  transformed_correctness = calculate_correctness(dataset, variants, src_tests, src_lang)
  logger.info('Testing transformed originals.')
  translated_correctness = calculate_correctness(dataset, translated_snippets, dst_tests, dst_lang)
  logger.info('Testing transformed variants.')
  transformed_translated_correctness = calculate_correctness(dataset, translated_variants, dst_tests, dst_lang)
  logger.info(f'\n'
              '========  Correctness  ========\n'
              f'Originals             : {original_correctness * 100:>6.2f}%\n'
              f'Variants              : {transformed_correctness * 100:>6.2f}%\n'
              f'Translated Originals  : {translated_correctness * 100:>6.2f}%\n'
              f'Translated Variants   : {transformed_translated_correctness * 100:>6.2f}%\n'
              f'===============================')


def evaluate_space(dataset: str, snippets: Sequence[Snippet], corpus: Sequence[Sequence[Snippet]], translated_snippets: Sequence[Snippet], translated_corpus: Sequence[Sequence[Snippet]], src_lang: str, dst_lang: str) -> None:
  logger.info(f'Evaluating {len(corpus)} sets of variants with {len(snippets)} snippets for each on {dataset}...')
  ids = tuple(snippet.id for snippet in snippets)
  src_tests = load_tests(dataset, src_lang, dst_lang, ids)[:len(snippets)]
  dst_tests = load_tests(dataset, src_lang, dst_lang, ids, translated=True)[:len(snippets)]
  logger.info('Testing originals.')
  original_correctness = calculate_correctness(dataset, snippets, src_tests, src_lang)
  logger.info('Testing variants.')
  transformed_correctness_list = [calculate_correctness(dataset, variants, src_tests, src_lang)
                                  for variants in tqdm(corpus, desc='Evaluating', total=len(corpus), leave=False)]
  logger.info('Testing transformed originals.')
  translated_correctness = calculate_correctness(dataset, translated_snippets, dst_tests, dst_lang)
  logger.info('Testing transformed variants.')
  transformed_translated_correctness_list = [calculate_correctness(dataset, variants, dst_tests, dst_lang)
                                             for variants in tqdm(translated_corpus, desc='Evaluating', total=len(corpus), leave=False)]
  logger.info(f'\n'
              '========  Correctness  ========\n'
              f'Originals             : {original_correctness * 100:>6.2f}%\n'
              f'Variants              : {sum(transformed_correctness_list) / len(transformed_correctness_list) * 100:>6.2f}%\n'
              f'Translated Originals  : {translated_correctness * 100:>6.2f}%\n'
              f'Translated Variants   : {sum(transformed_translated_correctness_list) / len(transformed_translated_correctness_list) * 100:>6.2f}%\n'
              f'===============================')
  df = pd.DataFrame({'variants': transformed_correctness_list, 'translated_variants': transformed_translated_correctness_list})
  df.to_csv(RESULT_DIR / f'{dataset}_correctness_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv', index=False)
