"""
Usage: python3 main.py [OPTIONS]...
"""

import argparse

from src.evaluate import evaluate
from src.extract import extract_source
from src.mutate import mutate_source
from src.translate import load_model, translate_with_model

DATASETS: list[str]
MODEL: str
SRC_LANG: str
DST_LANG: str
GPU: int = -1


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
                      choices=['deepseek-coder-7b-instruct-v1.5',
                               'Qwen2.5-Coder-1.5B-Instruct',
                               'Qwen2.5-Coder-3B-Instruct',
                               'Qwen2.5-Coder-7B-Instruct'],
                      required=True,
                      help='Specify the model to use.')
  parser.add_argument('--src-lang', default=default_src_lang, type=str,
                      choices=['java', 'cpp'],
                      help=f'Specify the source language, {default_src_lang} by default.')
  parser.add_argument('--dst-lang', default=default_dst_lang,type=str,
                      choices=['c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust'],
                      help=f'Specify the destination language, {default_dst_lang} by default.')
  parser.add_argument('-i', '--gpu-id', type=int, default=-1,
                      help='Specify the GPU to use.')
  args = parser.parse_args()
  global DATASETS, MODEL, SRC_LANG, DST_LANG, GPU
  DATASETS = args.dataset if args.dataset else default_datasets
  if args.model:
    MODEL = args.model
  if args.src_lang:
    SRC_LANG = args.src_lang
  if args.dst_lang:
    DST_LANG = args.dst_lang
  if args.gpu_id:
    GPU = args.gpu_id
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
    snippets = extract_source(dataset, SRC_LANG)[:1]
    mutants = mutate_source(snippets, SRC_LANG)
    translator = load_model(MODEL, gpu_id=GPU)
    translated_snippets = translate_with_model(snippets, translator, SRC_LANG, DST_LANG)
    translated_mutants = translate_with_model(mutants, translator, SRC_LANG, DST_LANG)
    evaluate(dataset, translated_snippets, translated_mutants, DST_LANG)


if __name__ == '__main__':
  main()
