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

from stylo_flora import Snippet
from stylo_flora.agent.base import BaseAgent, agent_factory
from stylo_flora.agent.tag_classifier import tag
from stylo_flora.agent.repairer import repair
from stylo_flora.agent.translator import translate
from stylo_flora.benchmarks import BaseBenchmark, benchmark_factory
from stylo_flora.logger import init_logger, logger
from stylo_flora.metrics.correctness import calc_correctness
from stylo_flora.metrics.f1_score import calc_macro_f1
from stylo_flora.transformer.base import BaseTransformer, transformer_factory


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Code task evaluation tool.'
                                               'All the datasets are evaluated by default.')
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model to use.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      choices=['code_translation', 'apr', 'code2tag', 'des_code2tag'],
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


def pick_snippets(snippets: Sequence[Snippet], args: argparse.Namespace, *, ensure_correct: bool = True) -> Sequence[Snippet]:
  # TODO: how to ensure correct transformation on problematic snippets?
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
    if not ensure_correct:
      return True
    return math.isclose(calc_correctness([snippet], lang=args.src_lang), 1.0)

  candidates = (i for i in indices if is_valid(snippets[i]))
  picked_indices = list(tqdm(islice(candidates, args.num_snippets),
                             desc='Picking snippets', total=args.num_snippets))
  picked_snippets = [snippets[i] for i in picked_indices]
  logger.verbose(f'Picked {len(picked_indices)} snippet indices: {picked_indices}')

  return picked_snippets


def cut_testcases(snippets: Sequence[Snippet], args: argparse.Namespace) -> None:
  if args.num_tests < 0:
    return
  for snippet in snippets:
    if len(snippet.args['testcases']) > args.num_tests:
      snippet.args['testcases'] = pd.Series(snippet.args['testcases']) \
        .sample(n=args.num_tests, random_state=args.seed).tolist()


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
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
  ) -> None:
  if not args.dst_lang:
    raise ValueError('Destination language must be specified for code translation task.')
  logger.info(f'Code translation task from {args.src_lang} to {args.dst_lang}.')

  snippets = benchmark.load_for_translation(args.src_lang, args.dst_lang)
  snippets = pick_snippets(snippets, args)
  cut_testcases(snippets, args)

  corpus = transformer.transform(
      snippets=snippets,
      lang=args.src_lang,
      seed=args.seed,
  )

  logger.info('Translating on originals.')
  res_orig = translate(agent, snippets, args.src_lang, args.dst_lang)
  logger.info('Translating on variants.')
  res_spanned = [translate(agent, variants, args.src_lang, args.dst_lang)
                       for variants in tqdm(corpus, desc='Translating',
                                            total=len(corpus), leave=False)]
  # TODO: serialize the result corpus as JSONL
  num_seq = len(corpus[0])

  logger.info('Evaluating correctness of code translation on the originals.')
  pass_res_orig = calc_correctness(res_orig, args.dst_lang)
  logger.info('Evaluating correctness of code translation on the variants.')
  pass_res_spanned = [calc_correctness([variants[i] for variants in res_spanned], args.dst_lang)
                      for i in tqdm(range(num_seq), desc='Evaluating', total=num_seq, leave=False)]
  logger.info(f'Correctness of {args.model} on {args.dataset}:\n'
              f'==  Correctness  ==\n'
              f'Originals : {pass_res_orig * 100:>6.2f}%\n'
              f'Variants  : {sum(pass_res_spanned) / len(pass_res_spanned) * 100:>6.2f}%\n'
              f'===================')

  df = pd.DataFrame({'res_orig': pass_res_orig, 'res_spanned': pass_res_spanned})
  save_results(f'correctness_{args.dataset}_{args.task}_{args.src_lang}_to_{args.dst_lang}_with_{args.model.replace("/", "-")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', df)

def evaluate_apr(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
  ) -> None:
  logger.info(f'APR task in {args.src_lang}.')

  snippets = benchmark.load_for_apr(args.src_lang)
  snippets = pick_snippets(snippets, args, ensure_correct=False)
  cut_testcases(snippets, args)

  corpus = transformer.transform(
      snippets=snippets,
      lang=args.src_lang,
      seed=args.seed,
      ensure_correct=False,
  )

  logger.info('Repairing on originals.')
  res_snippets = repair(agent, snippets, args.src_lang)
  logger.info('Repairing on variants.')
  res_corpus = [repair(agent, variants, args.src_lang)
                for variants in tqdm(corpus, desc='Repairing',
                                     total=len(corpus), leave=False)]
  num_seq = len(corpus[0])

  logger.info('Evaluating correctness of APR on the originals.')
  pass_res_orig = calc_correctness(res_snippets, args.src_lang)
  logger.info('Evaluating correctness of APR on the variants.')
  pass_res_spanned = [calc_correctness([variants[i] for variants in res_corpus], args.src_lang)
                      for i in tqdm(range(num_seq), desc='Evaluating', total=num_seq, leave=False)]
  logger.info(f'Correctness of {args.model} on {args.dataset}:\n'
              f'==  Correctness  ==\n'
              f'Originals : {pass_res_orig * 100:>6.2f}%\n'
              f'Variants  : {sum(pass_res_spanned) / len(pass_res_spanned) * 100:>6.2f}%\n'
              f'===================')

  df = pd.DataFrame({'res_orig': pass_res_orig, 'res_spanned': pass_res_spanned})
  save_results(f'correctness_{args.dataset}_{args.task}_{args.src_lang}_with_{args.model.replace("/", "-")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', df)


def _evaluate_tagging(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
    *,
    with_desc: bool,
  ) -> None:
  logger.info(f'Tag classification task in {args.src_lang}.')

  snippets = benchmark.load_for_tagging(args.src_lang)
  snippets = pick_snippets(snippets, args, ensure_correct=False)
  cut_testcases(snippets, args)

  corpus = transformer.transform(snippets=snippets,
      lang=args.src_lang,
      seed=args.seed,
      ensure_correct=False,
  )

  logger.info('Tagging on originals.')
  tags_orig = tag(agent, snippets, args.src_lang, with_desc)
  logger.info('Tagging on variants.')
  tags_spanned = [tag(agent, variants, args.src_lang, with_desc)
                for variants in tqdm(corpus, desc='Tagging',
                                     total=len(corpus), leave=False)]
  gloden_tags = [snippet.args['tags'] for snippet in snippets]
  num_seq = len(corpus[0])

  logger.info('Calculating F1 score of tag classification on the originals.')
  f1_res_orig = calc_macro_f1(tags_orig, gloden_tags)
  logger.info('Calculating F1 score of tag classification on the variants.')
  f1_res_spanned = [calc_macro_f1([variants[i] for variants in tags_spanned], gloden_tags)
                      for i in tqdm(range(num_seq), desc='Evaluating', total=num_seq, leave=False)]
  logger.info(f'F1 Score of {args.model} on {args.dataset}:\n'
              f'==  Macro F1 Score  ==\n'
              f'Originals :    {f1_res_orig * 100:>6.2f}%\n'
              f'Variants  :    {sum(f1_res_spanned) / len(f1_res_spanned) * 100:>6.2f}%\n'
              f'======================')

  df = pd.DataFrame({'res_orig': f1_res_orig, 'res_spanned': f1_res_spanned})
  save_results(f'f1_score_{args.dataset}_{args.task}_{args.src_lang}_with_{args.model.replace("/", "-")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', df)


def evaluate_code2tag(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
) -> None:
  _evaluate_tagging(benchmark, transformer, agent, args, with_desc=False)


def evaluate_des_code2tag(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
) -> None:
  _evaluate_tagging(benchmark, transformer, agent, args, with_desc=True)


def main():
  args = parse_args()
  init_logger(verbose=args.verbose, debug=args.debug)
  benchmark = benchmark_factory(args.dataset)
  transformer = transformer_factory()
  agent = agent_factory(args.model)
  logger.info(f'Evaluating {args.model} on {args.dataset} with transformer {transformer.__class__.__name__}...')

  evaluator = globals().get(f'evaluate_{args.task}')
  if not evaluator:
    raise ValueError(f'Unsupported task {args.task} for evaluation.')
  evaluator(benchmark, transformer, agent, args)


if __name__ == '__main__':
  main()
