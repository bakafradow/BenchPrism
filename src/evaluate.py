import json
import re
import subprocess
import tempfile
from typing import Sequence

from tqdm import tqdm

from . import Snippet
from .utils import extract_field_from, logger


class CompilationError(Exception):
  def __init__(self, message: str, stderr: str = ''):
    self.stderr = stderr
    super().__init__(message)


def _compile(code: str, lang: str) -> str:
  """
  Compiles the code snippet.
  :param code: the code snippet to be compiled
  :param lang: the language of the code snippet
  :return: the path of the compiled executable or the code itself for interpreted languages
  """
  match lang:
    case 'java':
      matched = re.search(r'public\s+(?:final\s+)?class\s+(\w+)', code)
      if not matched:
        raise CompilationError('Failed to extract class name from Java code.')
      class_name = matched.group(1)
      with tempfile.TemporaryDirectory() as tmpdir, open(f'{tmpdir}/{class_name}.java', 'w') as f:
        f.write(code)
        f.flush()
        executable = re.sub(r'\.java$', '', f.name)
        returned = subprocess.run(['javac', f.name], stderr=subprocess.PIPE)
        if returned.returncode != 0:
          raise CompilationError(f'Failed to compile {f.name}.', returned.stderr)
      return executable
    case 'cpp':
      with tempfile.NamedTemporaryFile(suffix='.cpp') as f:
        f.write(code.encode())
        f.flush()
        executable = re.sub(r'\.cpp$', '', f.name)
        returned = subprocess.run(['g++', f.name, '-o', executable], stderr=subprocess.PIPE)
        if returned.returncode != 0:
          raise CompilationError(f'Failed to compile {f.name}.', returned.stderr)
      return executable
    case 'python':
      return code
    case _:
      raise TypeError(f'Unsupported language: {lang}.')


def run_with_assertion(code: str, test: str, lang: str) -> bool:
  """
  Runs the code snippet with the test which asserts the correctness of the code.
  :param code: the code snippet to be tested WITHOUT main function
  :param test: the code snippet that contains the main function of the test
  :param lang: the language of the code snippet
  :return: whether the code snippet passes the test
  """
  try:
    executable = _compile(code + test, lang)
  except CompilationError as err:
    logger.warning(err)
    logger.verbose(err.stderr)
    return False
  match lang:
    case 'java':
      args = ['java']
    case 'cpp':
      args = []
    case 'python':
      args = ['python', '-c']
    case _:
      raise TypeError(f'Unsupported language: {lang}.')
  returned = subprocess.run(args + [executable], stderr=subprocess.PIPE)
  logger.verbose(f'Failed assertion:\n{returned.stderr}')
  return returned.returncode == 0


def run_with_io(code: str, test: list[dict], lang: str) -> bool:
  """
  Runs the code snippet with the test which checks the input-output behavior of the code.
  :param code: the code snippet to be tested with main function
  :param test: the input-output pair of the test
  :param lang: the language of the code snippet
  :return: whether the code snippet passes the test
  """
  try:
    executable = _compile(code, lang)
  except CompilationError as err:
    logger.warning(err)
    logger.verbose(err.stderr)
    return False
  match lang:
    case 'java':
      args = ['java']
    case 'cpp':
      args = []
    case 'python':
      args = ['python', '-c']
    case _:
      raise TypeError(f'Unsupported language: {lang}.')
  for pair in test:
    returned = subprocess.run(args + [executable], input=pair['input'], text=True, capture_output=True)
    if returned.returncode != 0:
      logger.verbose(f'Returned {returned.returncode} on input:\n{pair["input"].strip()}\nStandard Error:\n{returned.stderr}')
      return False
    if returned.stdout.strip() != pair['output'][0].strip():
      logger.verbose(f'Failed on input:\n{pair["input"].strip()}\nExpected:\n{pair["output"][0].strip()}\nActual:\n{returned.stdout.strip()}')
      return False
  return True


def calculate_correctness(dataset: str, snippets: Sequence[Snippet], tests: Sequence[str], lang: str) -> float:
  """
  Checks the correctness of the translated code with the tests.
  :param dataset: the dataset name
  :param snippets: the translated code snippets
  :param tests: the tests
  :param lang: the language of the code snippets
  """
  correct_count = 0
  for snippet, test in tqdm(zip(snippets, tests), desc='Evaluating', total=len(snippets), leave=False):
    if not snippet:
      continue
    match dataset:
      case 'HumanEvalX':
        if run_with_assertion(snippet.code, test, lang):
          correct_count += 1
      case 'xCodeEval':
        if run_with_io(snippet.code, test, lang):
          correct_count += 1
      case _:
        raise TypeError(f'Unsupported dataset: {dataset}.')
  return correct_count / len(snippets)


def load_tests(dataset: str, src_lang: str, dst_lang: str, *, translated: bool = False) -> Sequence[str]:
  match dataset:
    case 'HumanEvalX':
      tests = extract_field_from(dataset, dst_lang if translated else src_lang, 'test')
    case 'xCodeEval':
      with open('data/xCodeEval/unittest_db.json', 'r') as f:
        unittests = json.load(f)
      uids = extract_field_from(dataset, src_lang, 'src_uid')
      tests = [unittests[uid] for uid in uids]
      for test in tests:
        for pair in test:
          pair['input'] = pair['input'].replace('\r\n', '\n')
          pair['output'] = [line.replace('\r\n', '\n') for line in pair['output']]
    case 'CodeXGLUE':
      raise NotImplementedError('CodeXGLUE dataset does not provide tests.')
    case _:
      raise TypeError(f'Unknown dataset: {dataset}.')
  return tests


def evaluate(dataset: str, snippets: Sequence[Snippet], mutants: Sequence[Snippet], translated_snippets: Sequence[Snippet], translated_mutants: Sequence[Snippet], src_lang: str, dst_lang: str) -> None:
  """
  Evaluates the space spanned by the translated code relative to the original source code
  :param dataset: dataset name
  :param mutants: the original mutated code snippets
  :param translated_snippets: the translated original code snippets
  :param translated_mutants: the translated mutated code snippets
  :param src_lang: the source language of the code snippets
  :param dst_lang: the target language of the code snippets
  """
  logger.info(f'Evaluating {len(translated_snippets)} snippets on {dataset}...')
  if not len(mutants) == len(translated_snippets) == len(translated_mutants):
    raise ValueError('The number of snippets and mutants should equal.')
  src_tests = load_tests(dataset, src_lang, dst_lang)[:len(mutants)]
  dst_tests = load_tests(dataset, src_lang, dst_lang, translated=True)[:len(mutants)]
  original_correctness = calculate_correctness(dataset, snippets, src_tests, src_lang)
  mutated_correctness = calculate_correctness(dataset, mutants, src_tests, src_lang)
  translated_correctness = calculate_correctness(dataset, translated_snippets, dst_tests, dst_lang)
  mutated_translated_correctness = calculate_correctness(dataset, translated_mutants, dst_tests, dst_lang)
  logger.info(f'Original correctness: {original_correctness}.')
  logger.info(f'Mutated correctness: {mutated_correctness}.')
  logger.info(f'Translated correctness: {translated_correctness}.')
  logger.info(f'Mutated-translated correctness: {mutated_translated_correctness}.')
