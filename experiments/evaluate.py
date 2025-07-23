"""
Assesses the robustness of code task models by the following steps:

1. Extracts source code from different datasets into unified data structure.
2. Applies transformations to the source code to generate a set of transformed code with a code style transformer.
3. Performs code tasks with snippets with the evaluated model.
4. Evaluates the space spanned by the translated code relative to the original source code.
"""

import argparse
import math
import os
import random
from collections.abc import Sequence
from datetime import datetime
from itertools import islice
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from stylo_flora import Snippet, TestBatch
from stylo_flora.agent.base import BaseAgent, agent_factory
from stylo_flora.agent.classifier import classify
from stylo_flora.agent.repairer import repair
from stylo_flora.agent.translator import translate
from stylo_flora.benchmarks import BaseBenchmark, benchmark_factory
from stylo_flora.logger import init_logger, logger
from stylo_flora.metrics.correctness import calc_correctness
from stylo_flora.metrics.f1_score import calc_macro_f1
from stylo_flora.transformer.base import transformer_factory


TASKS_WITHOUT_CORRECTNESS = {'apr', 'tag_classification'}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Code task evaluation tool.'
                                               'All the datasets are evaluated by default.')
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model to use.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      help='Specify the code task to evaluate on.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('--dst-lang', type=str, required=False,
                      help='Specify the destination language. Only used for code translation task.')
  parser.add_argument('-n', '--num-snippets', type=int, default=-1,
                      help='Limit the number of snippets to test. -1 for all.')
  parser.add_argument('--num-tests', type=int, default=-1,
                      help='Limit the number of test cases for each snippet. -1 for all.')
  parser.add_argument('-r', '--random', action='store_true', default=False,
                      help='Select snippets randomly with the seed instead of sequentially.')
  parser.add_argument('--seed', type=int, default=42,
                      help='Set the random seed for reproducibility.')
  parser.add_argument('-v', '--verbose', action='store_true', default=False,
                      help='If set, enables verbose level logging.')
  parser.add_argument('--debug', action='store_true', default=False,
                      help='If set, enables debugging level logging.')
  args = parser.parse_args()
  return args


def pick_snippets(snippets: Sequence[Snippet], args: argparse.Namespace, benchmark: BaseBenchmark) -> Sequence[Snippet]:
  if args.num_snippets < 0:
    return snippets
  if not args.random:
    logger.verbose(f'Picking first {args.num_snippets} snippets sequentially.')
    return snippets[:args.num_snippets]

  indices = list(range(len(snippets)))
  random.seed(args.seed)
  random.shuffle(indices)

  # TODO: cache mechanism for correctness check
  def is_valid(snippet: Snippet) -> bool:
    if args.task in TASKS_WITHOUT_CORRECTNESS:
      return True
    test_batch = benchmark.load_tests([snippet.id])[0]
    if args.num_tests >= 0 and len(test_batch) > args.num_tests:
      test_batch = pd.Series(test_batch).sample(n=args.num_tests, random_state=args.seed).tolist()
    return math.isclose(calc_correctness([snippet], [test_batch], lang=args.src_lang), 1.0)

  candidates = (i for i in indices if is_valid(snippets[i]))
  picked_indices = list(tqdm(islice(candidates, args.num_snippets),
                             desc='Picking snippets', total=args.num_snippets))
  picked_snippets = [snippets[i] for i in picked_indices]
  logger.verbose(f'Picked {len(picked_indices)} snippet indices: {picked_indices}')

  return picked_snippets


def pick_test_batches(test_batches: Sequence[TestBatch], args: argparse.Namespace) -> Sequence[TestBatch]:
  if args.num_tests < 0:
    return test_batches
  picked_batches = []
  for batch in tqdm(test_batches, desc='Picking test batches', total=len(test_batches), leave=False):
    if len(batch) > args.num_tests:
      batch = pd.Series(batch).sample(n=args.num_tests, random_state=args.seed).tolist()
    picked_batches.append(batch)
  return picked_batches


def save_results(filename: str, df: pd.DataFrame) -> None:
  with open('settings.yml') as f:
    config = yaml.safe_load(f)['metrics']
  result_dir = Path(config['result_dir'])
  os.makedirs(result_dir, exist_ok=True)
  fallback_rate_path = result_dir / 'fallback_rates.csv'
  if os.path.exists(fallback_rate_path):
    df['fallback_rate'] = pd.read_csv(fallback_rate_path)['fallback_rate']
    os.remove(fallback_rate_path)
  df.to_csv(result_dir / filename, index=False)


def evaluate_code_translation(
    agent: BaseAgent,
    args: argparse.Namespace,
    snippets: Sequence[Snippet],
    corpus: Sequence[Sequence[Snippet]],
    test_batches: Sequence[TestBatch],
  ) -> None:
  res_orig = translate(agent, snippets, args.src_lang, args.dst_lang)
  res_spanned = [translate(agent, variants, args.src_lang, args.dst_lang)
                       for variants in tqdm(corpus, desc='Translating',
                                            total=len(corpus), leave=False)]
  # TODO: serialize the result corpus as JSONL
  num_seq = len(corpus[0])

  logger.info('Testing originals.')
  pass_orig = calc_correctness(snippets, test_batches, args.src_lang)
  logger.info('Testing variants.')
  pass_spanned = [calc_correctness([variants[i] for variants in corpus], test_batches, args.src_lang)
                for i in tqdm(range(num_seq), desc='Evaluating', total=num_seq, leave=False)]
  logger.info('Testing transformed originals.')
  pass_res_orig = calc_correctness(res_orig, test_batches, args.dst_lang)
  logger.info('Testing transformed variants.')
  pass_res_spanned = [calc_correctness([variants[i] for variants in res_spanned],
                                                 test_batches, args.dst_lang)
                           for i in tqdm(range(num_seq), desc='Evaluating', total=num_seq, leave=False)]
  logger.info(f'\n'
              f'Correctness of {args.model} on {args.dataset}:\n'
              f'========  Correctness  ========\n'
              f'Originals             : {pass_orig * 100:>6.2f}%\n'
              f'Variants              : {sum(pass_spanned) / len(pass_spanned) * 100:>6.2f}%\n'
              f'Translated Originals  : {pass_res_orig * 100:>6.2f}%\n'
              f'Translated Variants   : {sum(pass_res_spanned) / len(pass_res_spanned) * 100:>6.2f}%\n'
              f'===============================')

  df = pd.DataFrame({'res_orig': pass_res_orig, 'res_spanned': pass_res_spanned})
  save_results(f'correctness_{args.dataset}_{args.task}_{args.src_lang}_to_{args.dst_lang}_with_{args.model.replace("/", "-")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', df)

def evaluate_apr(
    agent: BaseAgent,
    args: argparse.Namespace,
    snippets: Sequence[Snippet],
    corpus: Sequence[Sequence[Snippet]],
    test_batches: Sequence[TestBatch],
  ) -> None:
  res_snippets = repair(agent, snippets, args.src_lang)
  res_corpus = [repair(agent, variants, args.src_lang)
                for variants in tqdm(corpus, desc='Repairing',
                                     total=len(corpus), leave=False)]
  num_seq = len(corpus[0])

  logger.info('Testing originals.')
  pass_res_orig = calc_correctness(res_snippets, test_batches, args.src_lang)
  logger.info('Testing variants.')
  pass_res_spanned = [calc_correctness([variants[i] for variants in res_corpus],
                                            test_batches, args.src_lang)
                      for i in tqdm(range(num_seq), desc='Evaluating', total=num_seq, leave=False)]
  logger.info(f'\n'
              f'Correctness of {args.model} on {args.dataset}:\n'
              f'========  Correctness  ========\n'
              f'Repaired Originals    : {pass_res_orig * 100:>6.2f}%\n'
              f'Repaired Variants     : {sum(pass_res_spanned) / len(pass_res_spanned) * 100:>6.2f}%\n'
              f'===============================')

  df = pd.DataFrame({'res_orig': pass_res_orig, 'res_spanned': pass_res_spanned})
  save_results(f'correctness_{args.dataset}_{args.task}_{args.src_lang}_with_{args.model.replace("/", "-")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', df)



def evaluate_tag_classification(
    agent: BaseAgent,
    args: argparse.Namespace,
    snippets: Sequence[Snippet],
    corpus: Sequence[Sequence[Snippet]],
  ) -> None:
  tags_orig = classify(agent, snippets, args.src_lang)
  tags_spanned = [classify(agent, variants, args.src_lang)
                for variants in tqdm(corpus, desc='Classifying',
                                     total=len(corpus), leave=False)]
  gloden_tags = [snippet.args['tags'] for snippet in snippets]
  num_seq = len(corpus[0])

  logger.info('Testing originals.')
  f1_res_orig = calc_macro_f1(tags_orig, gloden_tags)
  logger.info('Testing variants.')
  f1_res_spanned = [calc_macro_f1([variants[i] for variants in tags_spanned], gloden_tags)
                      for i in tqdm(range(num_seq), desc='Evaluating', total=num_seq, leave=False)]
  logger.info(f'\n'
              f'F1 Score of {args.model} on {args.dataset}:\n'
              f'=======  Macro F1 Score  ======\n'
              f'Classified Originals  : {f1_res_orig * 100:>6.2f}%\n'
              f'Classified Variants   : {sum(f1_res_spanned) / len(f1_res_spanned) * 100:>6.2f}%\n'
              f'===============================')

  df = pd.DataFrame({'res_orig': f1_res_orig, 'res_spanned': f1_res_spanned})
  save_results(f'correctness_{args.dataset}_{args.task}_{args.src_lang}_with_{args.model.replace("/", "-")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', df)


def main():
  args = parse_args()
  init_logger(verbose=args.verbose, debug=args.debug)
  benchmark = benchmark_factory(args.dataset)
  transformer = transformer_factory()
  agent = agent_factory(args.model)
  logger.info(f'Evaluating {args.model} on {args.dataset} with transformer {transformer.__class__.__name__}...')

  match args.task:
    case 'code_translation':
      if not args.dst_lang:
        raise ValueError('Destination language must be specified for code translation task.')
      logger.info(f'Code translation task from {args.src_lang} to {args.dst_lang}.')
      snippets = benchmark.load_for_translation(args.src_lang)
    case 'apr':
      logger.info(f'APR task in {args.src_lang}.')
      snippets = benchmark.load_for_apr(args.src_lang)
    case 'tag_classification':
      logger.info(f'Tag classification task in {args.src_lang}.')
      snippets = benchmark.load_for_tagging(args.src_lang)
    case _:
      raise ValueError(f'Unknown task {args.task}..')
  picked_snippets = pick_snippets(snippets, args, benchmark)

  # TODO: how to ensure correct transformation on problematic snippets?
  ids = (snippet.id for snippet in picked_snippets)
  test_batches = benchmark.load_tests(ids)
  picked_batches = pick_test_batches(test_batches, args)

  corpus = transformer.transform(
      snippets=picked_snippets,
      test_batches=picked_batches,
      lang=args.src_lang,
      seed=args.seed,
      ensure_correct=args.task not in TASKS_WITHOUT_CORRECTNESS,
  )

  match args.task:
    case 'code_translation':
      evaluate_code_translation(
          agent=agent,
          args=args,
          snippets=picked_snippets,
          corpus=corpus,
          test_batches=picked_batches,
      )
    case 'apr':
      evaluate_apr(
          agent=agent,
          args=args,
          snippets=picked_snippets,
          corpus=corpus,
          test_batches=picked_batches,
      )
    case 'tag_classification':
      evaluate_tag_classification(
          agent=agent,
          args=args,
          snippets=picked_snippets,
          corpus=corpus,
      )
    case _:
      raise ValueError(f'Unknown task {args.task}.')


if __name__ == '__main__':
  main()
