"""
The original ClassEval-T repository has several issues on file names and directory names, resulting in failures while evaluating. 🥱
"""

import re
import shutil
import sys
from pathlib import Path


def main():
  if len(sys.argv) < 2:
    print("Usage: python amend_dataset.py <dataset_dir>")
    exit(1)
  dataset_dir = Path(sys.argv[1])
  print(f'Amending ClassEval-T at {dataset_dir}...')

  try:
    shutil.move(dataset_dir / 'cpp' / 'test' / 'test_CalendarUti.cpp',
                dataset_dir / 'cpp' / 'test' / 'test_CalendarUtil.cpp')
  except FileNotFoundError:
    pass

  try:
    shutil.move(dataset_dir / 'java' / 'solutuon',
                dataset_dir / 'java' / 'solution')
  except FileNotFoundError:
    pass

  try:
    shutil.move(dataset_dir / 'py' / 'test' / 'DatabaseOperation.py',
                dataset_dir / 'py' / 'test' / 'DatabaseProcessor.py')
  except FileNotFoundError:
    pass

  try:
    path = dataset_dir / 'java' / 'test' / 'IpUtilTest.java'
    with open(path, 'r', encoding='utf-8') as f:
      content = f.read()
    content = content.replace('IPUtil', 'IpUtil')
    with open(path, 'w', encoding='utf-8') as f:
      f.write(content)
  except FileNotFoundError:
    pass

  try:
    path = dataset_dir / 'py' / 'test' / 'AccessGatewayFilter.py'
    with open(path, 'r', encoding='utf-8') as f:
      content = f.read()
    content = content.replace('meigimport', 'import')
    with open(path, 'w', encoding='utf-8') as f:
      f.write(content)
  except FileNotFoundError:
    pass

  try:
    path = dataset_dir / 'java' / 'solution' / 'Classroom.java'
    with open(path, 'r', encoding='utf-8') as f:
      content = f.read()
    content = content.replace('ClassroomManagementTest', 'ClassroomTest')
    with open(path, 'w', encoding='utf-8') as f:
      f.write(content)
  except FileNotFoundError:
    pass

  for file in (dataset_dir / 'py' / 'test').glob('*.py'):
    try:
      with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
      content = re.sub(r'from translation\.solution_py\.\w+ import \w+', '', content, flags=re.DOTALL)
      with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
    except FileNotFoundError:
      pass

  print('Done.')


if __name__ == '__main__':
  main()
