"""
Updates the format of candidate indices cached by `run.py` from an list to a map.
"""

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from scripts.experiment.run import SUPPORTED_TASKS
from scripts.experiment.utils import load_snippets


def main():
  with open(args.file, 'r') as f:
    candidates = json.load(f)
  print(f'Loaded {len(candidates)} valid indices from {args.file}.')

  valid_set = set(candidates)
  invalid_set = set(range(len(snippets))) - valid_set
  candidate_map = dict.fromkeys(candidates, True) | dict.fromkeys(invalid_set, False)

  shutil.move(args.file, args.file.with_name(args.file.name + '.bak'))
  with open(args.file, 'w') as f:
    json.dump(candidate_map, f)
  print('Done.')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset.')
  parser.add_argument('-f', '--file', type=Path, required=True,
                      help='Specify the path to the cached candidate file.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      choices=SUPPORTED_TASKS,
                      help='Specify the code task to evaluate on.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('--dst-lang', type=str, required=False,
                      help='Specify the destination language. Only used for code translation task.')
  args = parser.parse_args()
  snippets = load_snippets(args)
  main()
