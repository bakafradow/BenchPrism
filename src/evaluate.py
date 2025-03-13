from typing import Sequence
import subprocess
import tempfile

from datasets import load_dataset

from . import Snippet


def run_with_test(code: str, test: str, lang: str) -> bool:
  """
  Runs the code snippet with the test.
  :param code: the code snippet
  :param test: the test
  :param lang: the language of the code snippet
  :return: whether the code snippet passes the test
  """
  to_execute = code + test
  match lang:
    case 'cpp':
      # compile the code to a temporary file and run it
      with tempfile.NamedTemporaryFile(suffix='.cpp') as f:
        f.write(to_execute.encode())
        f.flush()
        returned = subprocess.run(['g++', f.name, '-o', f.name.replace('.cpp$', '')])
        if returned.returncode != 0:
          print(f'Failed to compile {f.name}.')
          return False
        returned = subprocess.run([f.name])
        return returned.returncode == 0
    case 'python':
      returned = subprocess.run(['python', '-c', to_execute])
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


def evaluate(dataset: str, snippets: Sequence[Snippet], mutants: Sequence[Snippet], lang: str) -> None:
  """
  Evaluates the space spanned by the translated code relative to the original source code
  :param dataset: dataset name
  :param snippets: the translated original code snippets
  :param mutations: the translated mutated code snippets
  :param lang: the language of the code snippets
  """
  print(f'Evaluating on {dataset}...')
  match dataset:
    case 'HumanEvalX':
      # load tests in corresponding language
      ds = load_dataset('THUDM/humaneval-x', lang, trust_remote_code=True)
      tests = [row['test'] for row in ds['test']]
      # evaluate the translated code with the tests, and print the results
      original_correctness = calculate_correctness(snippets, tests, lang)
      mutated_correctness = calculate_correctness(mutants, tests, lang)
    case _:
      raise TypeError(f'Unknown dataset: {dataset}.')
  print(f'Original correctness: {original_correctness}.')
  print(f'Mutated correctness: {mutated_correctness}.')
