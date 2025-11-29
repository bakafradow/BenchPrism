import re
from argparse import ArgumentParser
from pathlib import Path

import jsonlines

PATTERN = r'assert\s*\(?\s*f\s*\((.*)\)\s*(?:.\s*equals\s*\((.*)\)|==\s*(.*))\s*\)?'


def amend(output: str, group_idx) -> str:
  global count
  matched = re.search(PATTERN, output, re.DOTALL)
  if matched:
    count += 1
    return matched.group(group_idx)
  return output


def main():
  with jsonlines.open(args.file, 'r') as reader:
    data = list(reader)

  for row in data:
    if output := row.get('output'):
      row['output'] = amend(output, group_idx)
    if row.get('variant_outputs'):
      for i, output in enumerate(row['variant_outputs']):
        if output:
          row['variant_outputs'][i] = amend(output, group_idx)

  amended_file = args.file.with_stem('amended_' + args.file.stem)
  with jsonlines.open(amended_file, mode='w') as writer:
    writer.write_all(data)
  print(f'Amended {count} outputs, saved to {amended_file}.')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', '--file', type=Path, required=True,
                      help='Specify the path to the jsonl file that contains model outputs.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      choices=['input_reasoning', 'output_reasoning'],
                      help='Specify the reasoning type of task.')
  args = parser.parse_args()
  match args.task:
    case 'input_reasoning':
      group_idx = 1
    case 'output_reasoning':
      group_idx = 2
    case _:
      raise ValueError(f'Unknown task: {args.task}')

  count = 0
  main()
