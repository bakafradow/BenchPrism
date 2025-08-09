"""
Given a snippet dumped by EGSI, finds the simplest sequence that triggers failure and then reduces the snippet with Perses.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from tqdm import tqdm

from try_spanning import span


def reduce_seq(snippet: str, seq: list[int]) -> None:
  def worker(idx: int):
    selection = seq[idx]
    seq[idx] = -1
    mutant_str = span(snippet, seq)
    matched = re.search(r'public\s+(?:final\s+)?class\s+(\w+)', mutant_str)
    if not matched:
      print('No class name found. Skipping.')
      seq[idx] = selection
      return
    class_name = matched.group(1)
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = Path(temp_dir) / f'{class_name}.java'
      with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(mutant_str)
      returned = subprocess.run(['javac', str(temp_file)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
      if returned.returncode == 0:
        seq[idx] = selection

  with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(worker, range(len(seq))), desc='Reducing sequence', total=len(seq), leave=False))


def main():
  if len(sys.argv) < 2:
    print("Usage: python reduce_dumped.py <dump_dir>")
    exit(1)
  with open('settings.yml') as f:
    result_dir = Path(yaml.safe_load(f)['metrics']['result_dir'])
  dump_dir = result_dir / sys.argv[1]
  if not dump_dir.exists():
    print('Dump directory does not exist.')
    exit(1)
  reduce_dir = result_dir / f'{sys.argv[1]}_reduced'
  os.makedirs(reduce_dir, exist_ok=True)

  for file in tqdm(dump_dir.glob('*.txt'), desc='Processing files', leave=False, total=len(list(dump_dir.glob('*.txt')))):
    with open(file, 'r', encoding='utf-8') as f:
      snippet = f.read()
      print(f'Read snippet with {len(snippet.splitlines())} lines from {file}')
    matched = re.search(r'// Seq=(\[[^\]]*\])', snippet)
    if not matched:
      print(f'No sequence found in {file}. Skipping.')
      continue
    seq_str = matched.group(1)
    seq = ast.literal_eval(seq_str)

    reduce_seq(snippet, seq)

    if all(selection == -1 for selection in seq):
      print(f'All selections in {file} are -1. Skipping.')
      continue
    new_snippet = re.sub(r'// Seq=\[[^\]]*\]', f'// Seq={seq}', snippet, count=1)
    with open(reduce_dir / file.name, 'w', encoding='utf-8') as f:
      f.write(new_snippet)
      print(f'Wrote reduced snippet with {len(new_snippet.splitlines())} lines to {reduce_dir / file.name}')


if __name__ == '__main__':
  main()
