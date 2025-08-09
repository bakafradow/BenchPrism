"""
The original ClassEval-T repository has several issues on file names and directory names, resulting in failures while evaluating. 🥱
Specifically, this script executes the following operations:
- mv ClassEval_T/cpp/test/test_CalendarUti.cpp ClassEval_T/cpp/test/test_CalendarUtil.cpp
- mv ClassEval_T/java/solutuon ClassEval_T/java/solution
- mv ClassEval_T/py/test/DatabaseOperation.py ClassEval_T/py/test/DatabaseProcessor.py
- sed -i 's/meigimport/import/g' ClassEval_T/py/test/AccessGatewayFilter.py
"""

from pathlib import Path
import shutil
import sys


def main():
  if len(sys.argv) < 2:
    print("Usage: python amend_dataset.py <dataset_dir>")
    exit(1)
  dataset_dir = Path(sys.argv[1])
  print(f'Amending ClassEval-T at {dataset_dir}...')
  shutil.move(dataset_dir / 'cpp' / 'test' / 'test_CalendarUti.cpp',
              dataset_dir / 'cpp' / 'test' / 'test_CalendarUtil.cpp')
  shutil.move(dataset_dir / 'java' / 'solutuon',
              dataset_dir / 'java' / 'solution')
  shutil.move(dataset_dir / 'py' / 'test' / 'DatabaseOperation.py',
              dataset_dir / 'py' / 'test' / 'DatabaseProcessor.py')
  with open(dataset_dir / 'py' / 'test' / 'AccessGatewayFilter.py', 'r', encoding='utf-8') as f:
    content = f.read()
  content = content.replace('meigimport', 'import')
  with open(dataset_dir / 'py' / 'test' / 'AccessGatewayFilter.py', 'w', encoding='utf-8') as f:
    f.write(content)
  print('Done.')


if __name__ == '__main__':
  main()
