"""
Updates the IDs in variant/output file for CoderUJB defect detection from `task_id` to `task_id-bug_id`.
"""

import json
import jsonlines
from argparse import ArgumentParser
from pathlib import Path

from scripts.postprocessing.utils import save_with_backups


def main():
  id_map = {}
  with open('data/CoderUJB/datasets/data/task_defectdetection_bench_1111|2048.json', 'r') as f:
    ds = json.load(f)
  for row in ds['code_ujb_defectdetection']:
    id_map.setdefault(row['task_id'], f'{row["task_id"]}-{row["bug_id"]}')
  with jsonlines.open(args.file, mode='r') as reader:
    data = list(reader)
  assert all(row['id'] in id_map for row in data)
  for row in data:
    row['id'] = id_map[row['id']]
  save_with_backups(data, args.file)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', '--file', type=Path, required=True,
                      help='Specify the path to the cached variant/output file.')
  args = parser.parse_args()
  main()
