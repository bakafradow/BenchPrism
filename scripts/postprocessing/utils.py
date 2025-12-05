import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jsonlines

BACKUP_SUFFIX = '.bak'
BACKUP_LIMIT = 1000


def save_with_backups(data: Iterable[Any], path: os.PathLike):
  try:
    existing: list[Path] = []
    p = Path(path)
    while p.exists():
      if len(existing) > BACKUP_LIMIT:
        raise ValueError(f'Reached backup limit of {BACKUP_LIMIT}.')
      existing.append(p)
      p = p.with_name(p.name + BACKUP_SUFFIX)
    for p in reversed(existing):
      shutil.move(p, p.with_name(p.name + BACKUP_SUFFIX))
  except Exception as e:
    print(f'Failed to save to {path}: {e}', file=sys.stderr)
    path = os.path.join(os.path.dirname(path), 'amended_' + os.path.basename(path))
    print(f'Use default path instead: {path}', file=sys.stderr)
  with jsonlines.open(path, mode='w') as writer:
    writer.write_all(data)
