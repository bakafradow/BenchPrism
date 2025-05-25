"""
Usage: python3 main.py [OPTIONS]...
"""

import argparse
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from .dataset.extractor import extract_source
from .evaluator.evaluator import evaluate_space
from .transformer.transformer import transform_source
from .downstream.translator.translator import load_model, translate_with_model
from .logger import init_logger


def parse_args() -> argparse.Namespace:
  default_datasets = ['HumanEvalX', 'xCodeEval', 'XLCoST', 'CodeXGLUE', 'G-TransEval', 'CodeNet']
  default_src_lang = 'java'
  default_dst_lang = 'cpp'
  parser = argparse.ArgumentParser(description='Code translation evaluation tool.'
                                               'All the datasets are evaluated by default.')
  parser.add_argument('-d', '--dataset', nargs=1, type=str,
                      choices=default_datasets,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str,
                      required=True,
                      help='Specify the model to use.')
  parser.add_argument('--src-lang', default=default_src_lang, type=str,
                      choices=['java', 'cpp'],
                      help=f'Specify the source language, {default_src_lang} by default.')
  parser.add_argument('--dst-lang', default=default_dst_lang, type=str,
                      choices=['c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust'],
                      help=f'Specify the destination language, {default_dst_lang} by default.')
  parser.add_argument('-i', '--gpu-id', type=int, default=-1,
                      help='Specify the GPU to use.')
  parser.add_argument('-n', '--num-snippets', type=int, default=-1,
                      help='Limit the number of snippets to test.')
  parser.add_argument('--seed', type=int, default=42,
                      help='Set the random seed for reproducibility.')
  parser.add_argument('--verbose', action='store_true', default=False,
                      help='If set, enables verbose level logging.')
  parser.add_argument('--debug', action='store_true', default=False,
                      help='If set, enables debugging level logging.')
  args = parser.parse_args()
  args.datasets = args.dataset if args.dataset else default_datasets
  return args


def main():
  """
  Assesses the robustness of code translation models by the following steps:

  1. Extracts source code from different datasets into unified data structure.

  2. Applies transformations to the source code to generate a set of transformed code with a code style transformer.

  3. Translates snippets in code set with code translation model.

  4. Evaluates the space spanned by the translated code relative to the original source code.
  """
  args = parse_args()
  init_logger(verbose=args.verbose, debug=args.debug)
  for dataset in args.datasets:
    snippets = extract_source(dataset, args.src_lang, args.dst_lang)
    if args.num_snippets >= 0:
      snippets = snippets[:args.num_snippets]
    corpus = transform_source(snippets, args.src_lang, seed=args.seed)
    translator = load_model(args.model, gpu_id=args.gpu_id)
    translated_snippets = translate_with_model(translator, snippets, args.src_lang, args.dst_lang)
    translated_corpus = [translate_with_model(translator, variants, args.src_lang, args.dst_lang)
                         for variants in tqdm(corpus, desc='Translating corpus', total=len(corpus), leave=False)]
    evaluate_space(dataset, snippets, corpus, translated_snippets, translated_corpus, args.src_lang, args.dst_lang)


if __name__ == '__main__':
  main()
