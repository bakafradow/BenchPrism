"""
The original TestBench repository has slight inconsistencies in metadata, resulting in potential bugs while evaluating.
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

import javalang


def extract_class_name(row: dict[str, str]) -> str | None:
  try:
    tree = javalang.parse.parse(row['full_context'])
  except javalang.tokenizer.LexerError:
    print(f'Failed to tokenize {row["relative_path"]}.')
    return None
  except javalang.parser.JavaSyntaxError:
    print(f'Failed to parse {row["relative_path"]}.')
    return None
  for node in tree.types:
    if (isinstance(node, javalang.tree.ClassDeclaration) and
        row['method_name'] in [node.name for node in node.body
                               if isinstance(node, javalang.tree.MethodDeclaration)]):
      return node.name
  return None


def main():
  print(f'Amending TestBench at {dataset_dir}...')

  count = 0
  files = list((dataset_dir / 'source_file_parser').iterdir())
  for file in tqdm(files, desc='Amending', total=len(files), leave=False):
    if file.suffix != '.json':
      continue
    with open(file, 'r') as f:
      data = json.load(f)
    for row in tqdm(data, desc=file.stem, total=len(data), leave=False):
      class_name = extract_class_name(row)
      if class_name and class_name != row['class_name']:
        row['class_name'] = class_name
        count += 1
    with open(file, 'w') as f:
      json.dump(data, f, indent=4)

  print(f'Done. Updated {count} class names.')


if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Usage: python amend_dataset.py <dataset_dir>")
    exit(1)
  dataset_dir = Path(sys.argv[1])
  main()
