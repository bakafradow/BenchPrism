import csv
import logging
import os
import sys
import tempfile
from collections import defaultdict

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
)

LANGUAGES = frozenset(('Java', 'C++', 'Python'))
LANG_TO_SUFFIX = {
  'Java': 'java',
  'C++': 'cpp',
  'Python': 'py',
}

START = 0
END = 1000
CODENET_DIR = os.path.join(os.path.dirname(__file__), '../../data/Project_CodeNet')


def is_accepted(accuracy: str) -> bool:
  if not accuracy:
    return False
  try:
    numer, denom = accuracy.split('/')
    return int(numer) == int(denom)
  except Exception as e:
    logging.warning(f'Error evaluating accuracy "{accuracy}": {e}')
    return False


if __name__ == '__main__':
  if len(sys.argv) != 2:
    logging.error('Usage: python extract.py <tarfile>')
    sys.exit(1)

  target_dict = defaultdict(list)
  for num in range(START, END):
    pid = f'p{num:05d}'
    for lang in LANGUAGES:
      metafile = os.path.join(CODENET_DIR, f'Project_CodeNet/metadata/{pid}.csv')
      if not os.path.exists(metafile):
        logging.warning(f'Skipping {metafile} as it does not exist.')
        continue
      with open(metafile, 'r') as f:
        reader = csv.DictReader(f)
        try:
          sid = next(row['submission_id'] for row in reader if row['language'] == lang and is_accepted(row['accuracy']))
        except StopIteration:
          logging.warning(f'No valid submission found for {lang} in {os.path.basename(metafile)}.')
          continue
      target_dict[pid].append((lang, sid))
  target_dict = {k: v for k, v in target_dict.items() if len(v) == len(LANGUAGES)}
  logging.info(f'Found {len(target_dict)} valid submissions.')

  with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
    f.write('\n'.join(f'Project_CodeNet/data/{pid}/{lang}/{sid}.{LANG_TO_SUFFIX[lang]}' for pid, pairs in target_dict.items() for lang, sid in pairs))
  os.system(f'pv {sys.argv[1]} | tar -xzvf - -C {CODENET_DIR} --files-from={f.name}')
  os.remove(f.name)
  logging.info('Extraction completed.')
