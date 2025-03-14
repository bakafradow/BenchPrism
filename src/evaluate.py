import re
import subprocess
import tempfile
from typing import Sequence

from . import Snippet
from .utils import extract_field_from


def run_with_test(code: str, test: str, lang: str) -> bool:
  """
  Runs the code snippet with the test.
  :param code: the code snippet
  :param test: the test
  :param lang: the language of the code snippet
  :return: whether the code snippet passes the test
  """
  program = code + test
  match lang:
    case 'cpp':
      # compile the code to a temporary file and run it
      with tempfile.NamedTemporaryFile(suffix='.cpp') as f:
        f.write(program.encode())
        f.flush()
        executable = re.sub(r'\.cpp$', '', f.name)
        returned = subprocess.run(['g++', f.name, '-o', executable])
        if returned.returncode != 0:
          print(f'Failed to compile {f.name}.')
          return False
        returned = subprocess.run([executable])
        return returned.returncode == 0
    case 'python':
      returned = subprocess.run(['python', '-c', program])
      return returned.returncode == 0
    case _:
      raise TypeError(f'Unsupported language: {lang}.')


def calculate_correctness(snippets: Sequence[Snippet], tests: Sequence[str], lang: str) -> float:
  """
  Checks the correctness of the translated code with the tests.
  :param snippets: the translated code snippets
  :param tests: the tests
  :param lang: the language of the code snippets
  """
  if len(snippets) != len(tests):
    raise ValueError('The number of snippets and tests should equal.')
  correct_count = 0
  for snippet, test in zip(snippets, tests):
    if run_with_test(snippet.code, test, lang):
      correct_count += 1
  return correct_count / len(snippets)


def load_tests(dataset: str, lang: str) -> Sequence[str]:
  match dataset:
    case 'HumanEvalX':
      tests = extract_field_from(dataset, lang, 'test')
    case _:
      raise TypeError(f'Unknown dataset: {dataset}.')
  return tests


def evaluate(dataset: str, snippets: Sequence[Snippet], mutants: Sequence[Snippet], lang: str) -> None:
  """
  Evaluates the space spanned by the translated code relative to the original source code
  :param dataset: dataset name
  :param snippets: the translated original code snippets
  :param mutations: the translated mutated code snippets
  :param lang: the language of the code snippets
  """
  print(f'Evaluating on {dataset}...')
  tests = load_tests(dataset, lang)[:3]
  original_correctness = calculate_correctness(snippets, tests, lang)
  mutated_correctness = calculate_correctness(mutants, tests, lang)
  print(f'Original correctness: {original_correctness}.')
  print(f'Mutated correctness: {mutated_correctness}.')
