import argparse
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from stylo_flora import Snippet, TestBatch
from stylo_flora.agent.base import BaseAgent, agent_factory
from stylo_flora.agent.translator import translate
from stylo_flora.benchmarks import benchmark_factory
from stylo_flora.logger import init_logger, logger
from stylo_flora.metrics.correctness import calculate_correctness
from stylo_flora.transformer.base import transformer_factory


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Code translation evaluation tool.'
                                               'All the datasets are evaluated by default.')
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model to use.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('--dst-lang', type=str, required=True,
                      help='Specify the destination language.')
  parser.add_argument('-n', '--num-snippets', type=int, default=-1,
                      help='Limit the number of snippets to test. -1 for all.')
  parser.add_argument('--seed', type=int, default=42,
                      help='Set the random seed for reproducibility.')
  parser.add_argument('-v', '--verbose', action='store_true', default=False,
                      help='If set, enables verbose level logging.')
  parser.add_argument('--debug', action='store_true', default=False,
                      help='If set, enables debugging level logging.')
  args = parser.parse_args()
  return args


def evaluate_translation(
    translator: BaseAgent,
    snippets: Sequence[Snippet],
    corpus: Sequence[Sequence[Snippet]],
    test_batches: Sequence[TestBatch],
    args: argparse.Namespace,
  ) -> None:
  translated_snippets = translate(translator, snippets, args.src_lang, args.dst_lang)
  translated_corpus = [translate(translator, variants, args.src_lang, args.dst_lang)
                       for variants in tqdm(corpus, desc='Translating corpus',
                                            total=len(corpus), leave=False)]

  logger.info('Testing originals.')
  res_original = calculate_correctness(snippets, test_batches, args.src_lang)
  logger.info('Testing variants.')
  res_varied = [calculate_correctness(variants, test_batches, args.src_lang)
                for variants in tqdm(corpus, desc='Evaluating', total=len(corpus), leave=False)]
  logger.info('Testing transformed originals.')
  res_translated = calculate_correctness(translated_snippets, test_batches, args.dst_lang)
  logger.info('Testing transformed variants.')
  res_translated_varied = [calculate_correctness(variants, test_batches, args.dst_lang)
                           for variants in tqdm(translated_corpus, desc='Evaluating', total=len(corpus), leave=False)]
  logger.info(f'\n'
              f'Correctness of {args.model} on {args.dataset}:\n'
              f'========  Correctness  ========\n'
              f'Originals             : {res_original * 100:>6.2f}%\n'
              f'Variants              : {sum(res_varied) / len(res_varied) * 100:>6.2f}%\n'
              f'Translated Originals  : {res_translated * 100:>6.2f}%\n'
              f'Translated Variants   : {sum(res_translated_varied) / len(res_translated_varied) * 100:>6.2f}%\n'
              f'===============================')

  with open('settings.yml') as f:
    config = yaml.safe_load(f)['metrics']
  result_dir = Path(config['result_dir'])
  os.makedirs(result_dir, exist_ok=True)
  df = pd.DataFrame({'translated': res_translated,
                     'translated_variants': res_translated_varied,
                     })
  fallback_rate_path = result_dir / 'fallback_rates.csv'
  if os.path.exists(fallback_rate_path):
    df['fallback_rate'] = pd.read_csv(fallback_rate_path)['fallback_rate']
    os.remove(fallback_rate_path)
  df.to_csv(result_dir / f'correctness_{args.dataset}_{args.src_lang}_to_{args.dst_lang}_with_{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)


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
  benchmark = benchmark_factory(args.dataset)
  transformer = transformer_factory()
  translator = agent_factory(args.model)
  logger.info(f'Evaluating {args.model} on {args.dataset} with transformer {transformer.__class__.__name__}...')

  snippets = benchmark.load_source(args.src_lang)
  if args.num_snippets >= 0:
    snippets = snippets[:args.num_snippets]
  ids = (snippet.id for snippet in snippets)
  test_batches = benchmark.load_tests(ids)
  corpus = transformer.transform(
      snippets=snippets,
      test_batches=test_batches,
      lang=args.src_lang,
      seed=args.seed,
  )

  evaluate_translation(
    translator=translator,
    snippets=snippets,
    corpus=corpus,
    test_batches=test_batches,
    args=args,
  )


if __name__ == '__main__':
  main()
