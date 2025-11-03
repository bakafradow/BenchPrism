"""
Given a snippet dumped by StyleX, finds the simplest sequence that triggers failure and then reduces the snippet with Perses.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from scripts.stylex.test_apply_by_seq import span_single, stylex


def reduce_seq(lang: str, snippet: str, seq: list[int]) -> None:
  def worker(idx: int):
    selection = seq[idx]
    seq[idx] = -1
    choice_dict = stylex._create_choice_dict(seq)
    mutant_str = span_single(snippet, lang, choice_dict)
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
  if len(sys.argv) < 4:
    print(f'Usage: python {os.path.basename(__file__)} <lang> <working_dir> <result_dir>')
    exit(1)
  working_dir = Path(sys.argv[2])
  if not working_dir.exists():
    print('Dump directory does not exist.')
    exit(1)
  result_dir = Path(sys.argv[3])
  os.makedirs(result_dir, exist_ok=True)

  for file in tqdm(working_dir.glob('*.txt'), desc='Processing files', leave=False, total=len(list(working_dir.glob('*.txt')))):
    with open(file, 'r', encoding='utf-8') as f:
      snippet = f.read()
      print(f'Read snippet with {len(snippet.splitlines())} lines from {file}')
    matched = re.search(r'// Seq=(\[[^\]]*\])', snippet)
    if not matched:
      print(f'No sequence found in {file}. Skipping.')
      continue
    seq_str = matched.group(1)
    seq = ast.literal_eval(seq_str)

    reduce_seq(sys.argv[1], snippet, seq)

    if all(selection == -1 for selection in seq):
      print(f'All selections in {file} are -1. Skipping.')
      continue
    new_snippet = re.sub(r'// Seq=\[[^\]]*\]', f'// Seq={seq}', snippet, count=1)
    with open(result_dir / file.name, 'w', encoding='utf-8') as f:
      f.write(new_snippet)
      print(f'Wrote reduced snippet with {len(new_snippet.splitlines())} lines to {result_dir / file.name}')


if __name__ == '__main__':
  main()
