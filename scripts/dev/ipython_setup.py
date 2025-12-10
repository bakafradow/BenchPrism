"""
Prepares IPython enviromnment for fine-grained debugging of transformations.

Usage:
  ipython -i <path_to_this_file> -- <args>
"""

import ast
import importlib.util
import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from openai import OpenAI  # type: ignore[attr-defined]
from tqdm import tqdm

from scripts.experiment.run import SUPPORTED_TASKS
from scripts.experiment.utils import load_snippets
from stylo_flora.logger import init_logger
from stylo_flora.transformer.stylex import StyleX

SRC_PATH = Path('~/playground/research/samples/src.java').expanduser()
CHOICE_DICT_PATH = Path('~/playground/research/samples/choices.json').expanduser()


def import_from_path(path: os.PathLike) -> Any:
  module_name = Path(path).stem
  spec = importlib.util.spec_from_file_location(module_name, path)
  if not spec or not spec.loader:
    raise ValueError(f'Failed load module spec from {path}.')
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def dump_code(idx: int) -> None:
  with open(SRC_PATH, 'w') as f:
    f.write(snippets[idx].data['code'])
  print('Dumped code to', SRC_PATH)


def dump_seq_as_dict() -> None:
  with open(CHOICE_DICT_PATH, 'w') as f:
    seq = ast.literal_eval(input('Input seq: '))
    choice_dict = stylex._create_choice_dict(seq)
    json.dump(choice_dict, f)
  print('Dumped choice dict to', CHOICE_DICT_PATH)


def format_timestamp(d: dict) -> dict:
  for k, v in d.items():
      if isinstance(v, int) and datetime.fromtimestamp(v).year in range(1900, 2100):
          d[k] = datetime.fromtimestamp(v).strftime('%Y-%m-%d %H:%M:%S')
  return d


def count_processable() -> tuple[int, int]:
  return sum(tqdm(map(stylex.is_processable, snippets), desc='Counting', total=len(snippets), leave=False)), len(snippets)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      choices=SUPPORTED_TASKS,
                      help='Specify the code task to evaluate on.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('--dst-lang', type=str, required=False,
                      help='Specify the destination language. Only used for code translation task.')
  args = parser.parse_args()

  init_logger(verbose=True)
  stylex = StyleX(lang=args.src_lang)
  snippets = load_snippets(args)
