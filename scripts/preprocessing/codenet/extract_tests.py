import os

import jsonlines

from bs4 import BeautifulSoup
from tqdm import tqdm


def extract_tests(file_path: str) -> list:
  """
  Extracts all sample test cases from a given HTML file containing problem descriptions.
  """
  with open(file_path, 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'lxml')

  test_cases = []
  suffixes = [''] + [' ' + str(i) for i in range(1, 10)] + [str(i) for i in range(1, 10)]
  labels = ['h', 'h2', 'h3', 'h4', 'H', 'H2', 'H3', 'H4']

  for suffix in suffixes:
    input_strings = [s + suffix for s in ('Sample Input', '入力', '入力例')]
    output_strings = [s + suffix for s in ('Sample Output', 'Sample Output for Input', 'Output for Sample Input', 'Output for the Sample Input', '出力', '出力例')]
    sample_input = soup.find(labels, string=input_strings)
    sample_output = soup.find(labels, string=output_strings)
    if not sample_input or not sample_output or \
       not sample_input.find_next_sibling('pre') or not sample_output.find_next_sibling('pre'):
      continue
    input_text = sample_input.find_next_sibling('pre').text.strip()
    output_text = sample_output.find_next_sibling('pre').text.strip()
    test_cases.append([input_text, output_text])

  if not test_cases:
    print(f'No test cases found in {file_path}.')
  return test_cases


CODENET_DIR = os.path.join(os.path.dirname(__file__), '../../data/Project_CodeNet')
DESC_DIR = os.path.join(CODENET_DIR, 'Project_CodeNet/problem_descriptions')


if __name__ == '__main__':
  with jsonlines.open(os.path.join(CODENET_DIR, 'Project_CodeNet/tests.jsonl'), 'w') as writer:
    for filename in tqdm(os.listdir(DESC_DIR), desc='Generating', total=len(os.listdir(DESC_DIR)), leave=False):
      if not filename.endswith('.html'):
        continue
      id = filename.split('.')[0]
      file_path = os.path.join(DESC_DIR, filename)
      tests = extract_tests(file_path)
      writer.write({
          'id': id,
          'test': tests
      })
