import os
import sys

import jsonlines

from bs4 import BeautifulSoup
from tqdm import tqdm


def get_test_cases(file_path: str) -> list:
  """解析 HTML 文件并提取多个 Sample Input 和 Sample Output"""
  with open(file_path, 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'lxml')

  # 提取所有的 Sample Input 和 Sample Output
  test_cases = []
  suffixes = [''] + [' ' + str(i) for i in range(1, 10)] + [str(i) for i in range(1, 10)]
  labels = ['h', 'h2', 'h3', 'h4', 'H', 'H2', 'H3', 'H4']
  # 查找所有的 Sample Input 和 Sample Output
  for suffix in suffixes:
    input_strings = [s + suffix for s in ['Sample Input', '入力', '入力例']]
    output_strings = [s + suffix for s in ['Sample Output', 'Sample Output for Input', 'Output for Sample Input', 'Output for the Sample Input', '出力', '出力例']]
    sample_input = soup.find(labels, string=input_strings)
    sample_output = soup.find(labels, string=output_strings)
    if not sample_input or not sample_output or \
       not sample_input.find_next_sibling('pre') or not sample_output.find_next_sibling('pre'):
      continue
    input_text = sample_input.find_next_sibling('pre').text.strip()
    output_text = sample_output.find_next_sibling('pre').text.strip()

    test_cases.append([input_text, output_text])
  # 返回每个数据集的输入和输出
  if not test_cases:
    print(f'No test cases found in {file_path}.')
  return test_cases


if __name__ == '__main__':
  directory = 'data/Project_CodeNet/problem_descriptions'
  with jsonlines.open('data/CodeNet/codenet_test.jsonl', 'w') as writer:
    for filename in tqdm(os.listdir(directory), desc='Generating', total=len(os.listdir(directory))):
      if not filename.endswith('.html'):
        continue
      id = filename.split('.')[0]
      file_path = os.path.join(directory, filename)
      test_cases = get_test_cases(file_path)
      writer.write({
          'id': id,
          'test': test_cases
      })
