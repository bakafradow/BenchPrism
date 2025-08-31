"""
The original ClassEval-T repository has several issues on file names and directory names, resulting in failures while evaluating. 🥱
Specifically, this script executes the following operations:
- mv ClassEval_T/cpp/test/test_CalendarUti.cpp ClassEval_T/cpp/test/test_CalendarUtil.cpp
- mv ClassEval_T/java/solutuon ClassEval_T/java/solution
- mv ClassEval_T/py/test/DatabaseOperation.py ClassEval_T/py/test/DatabaseProcessor.py
- sed -i 's/meigimport/import/g' ClassEval_T/py/test/AccessGatewayFilter.py
- sed -i 's/ClassroomManagementTest/ClassroomTest/g' ClassEval_T/java/solution/Classroom.java
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
    with open(dataset_dir / 'py' / 'test' / 'AccessGatewayFilter.py', 'r', encoding='utf-8') as f:
      content = f.read()
    content = content.replace('meigimport', 'import')
    with open(dataset_dir / 'py' / 'test' / 'AccessGatewayFilter.py', 'w', encoding='utf-8') as f:
      f.write(content)
  except FileNotFoundError:
    pass

  try:
    with open(dataset_dir / 'java' / 'solution' / 'Classroom.java', 'r', encoding='utf-8') as f:
      content = f.read()
    content = content.replace('ClassroomManagementTest', 'ClassroomTest')
    with open(dataset_dir / 'java' / 'solution' / 'Classroom.java', 'w', encoding='utf-8') as f:
      f.write(content)
  except FileNotFoundError:
    pass

  print('Done.')


if __name__ == '__main__':
  main()
