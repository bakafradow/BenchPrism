import json
import re
from argparse import ArgumentParser
from pathlib import Path

import jsonlines

from scripts.postprocessing import utils
from benchprism import IOTestCase

PATTERN_MULTILINE = re.compile(r'Input:\s*(?:```\s*(.*?)\s*```|([^\n]+)|\n((?:\w+\s*=\s*[^\n]+\s*)+))\n+Output:\s*(?:```\s*(.*?)\s*```|([^\n]+)|\n((?:\w+\s*=\s*[^\n]+\s*)+))', re.S)
PATTERN_SINGLELINE = re.compile(r'Input:\s*([^\n]*?),?\s*Output:\s*([^\n]*)', re.M)
PATTERN_ASSIGNMENT = re.compile(r'\w+\s*=\s*([^\n,]+),?', re.S)
PATTERN_QUOTED_LIST = re.compile(r'(?P<quo>[\'"`])(\[[^\]]+\])(?P=quo)', re.S)
PATTERN_QUOTED_STR = re.compile(r'(?P<quo>[\'"`])(.*?)(?P=quo)', re.S)


def _extract(string: str) -> str:
  string = PATTERN_ASSIGNMENT.sub(r'\1 ', string)
  list_str = PATTERN_QUOTED_LIST.search(string)
  single = json.loads(list_str.group(2))[0] if list_str else string
  single = PATTERN_QUOTED_STR.sub(r'\2', single)
  return single.encode('utf-8').decode('unicode-escape')


def amend(output: list[IOTestCase] | str) -> list[IOTestCase] | str:
  if not isinstance(output, str):
    return output
  global count
  testcases: list[IOTestCase] = []
  for match in PATTERN_MULTILINE.findall(output):
    input_str, output_str = match[0] or match[1] or match[2], match[3] or match[4] or match[5]
    input_ = _extract(input_str)
    output_ = _extract(output_str)
    testcases.append(IOTestCase.from_list([input_, output_]))
  for match in PATTERN_SINGLELINE.findall(output):
    input_str, output_str = match
    input_ = _extract(input_str)
    output_ = _extract(output_str)
    testcases.append(IOTestCase.from_list([input_, output_]))
  if testcases:
    count += 1
    return testcases
  return output


def main():
  with jsonlines.open(args.file, 'r') as reader:
    data = list(reader)

  for row in data:
    if output := row.get('output'):
      row['output'] = amend(output)
    if row.get('variant_outputs'):
      for i, output in enumerate(row['variant_outputs']):
        if output:
          row['variant_outputs'][i] = amend(output)

  utils.save_with_backups(data, args.file)
  print(f'Amended {count} outputs, saved to {args.file}.')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', '--file', type=Path, required=True,
                      help='Specify the path to the jsonl file that contains model outputs.')
  args = parser.parse_args()

  count = 0
  main()
