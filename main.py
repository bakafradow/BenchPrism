"""
Usage: python3 main.py [OPTIONS]...
"""

import argparse
from pprint import pprint
from tqdm import tqdm
from typing import List
from extract_source import extract
from transform import transform_exhaustively
from translate import translate_with_model
from evaluators.evaluate import evaluate

DATASETS: List[str]
MODEL: str
SRC_LANG: str
DEST_LANG: str


def parse_args():
  default_datasets = ['HumanEvalX', 'xCodeEval', 'XLCoST', 'CodeXGLUE', 'G-TransEval']
  default_src_lang = 'java'
  default_dst_lang = 'cpp'
  parser = argparse.ArgumentParser(description='Code translation evaluation tool.'
                                               'All the datasets are evaluated by default.')
  parser.add_argument('-d', '--dataset', nargs=1, type=str,
                      choices=default_datasets,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str,
                      choices=['CodeLamma'], required=True,
                      help='Specify the model to use.')
  parser.add_argument('--src-lang', default=default_src_lang, type=str,
                      choices=['java'],
                      help=f'Specify the source language, {default_src_lang} by default.')
  parser.add_argument('--dst-lang', default=default_dst_lang,type=str,
                      choices=['c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust'],
                      help=f'Specify the destination language, {default_dst_lang} by default.')
  args = parser.parse_args()
  global DATASETS, MODEL, SRC_LANG, DEST_LANG
  DATASETS = args.dataset if args.dataset else default_datasets
  if args.model:
    MODEL = args.model
  if args.src_lang:
    SRC_LANG = args.src_lang
  if args.dst_lang:
    DEST_LANG = args.dst_lang
  return args


def main():
  """
  Assesses the robustness of code translation models by the following steps:

  1. Extracts source code from different datasets into unified data structure.

  2. Applies transformations to the source code to generate a set of mutated code with a code style transformer.

  3. Translates snippets in code set with code translation model.

  4. Evaluates the space spanned by the translated code relative to the original source code.
  """
  parse_args()
  for dataset in DATASETS:
    source_code = extract(dataset, SRC_LANG)
    mutated_set = transform_exhaustively(source_code)
    translated_set = translate_with_model(mutated_set, MODEL, SRC_LANG, DEST_LANG)
    evaluate(source_code, translated_set)


if __name__ == '__main__':
  main()
