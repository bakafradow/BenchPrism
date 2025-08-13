"""
Assesses the robustness of code task models by the following steps:

1. Extracts source code from different datasets into unified data structure.
2. Applies transformations to the source code to generate a set of transformed code with a code style transformer.
3. Performs code tasks with snippets with the evaluated model.
4. Evaluates the space span by the translated code relative to the original source code.
"""

import argparse
import json
import math
import os
import random
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import islice
from operator import itemgetter
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from stylo_flora.transformer.base import BaseTransformer, transformer_factory
from stylo_flora.metrics import (calc_bertscore, calc_bleu, calc_codebleu,
                                 calc_correctness, calc_macro_f1, calc_meteor,
                                 calc_rouge)
from stylo_flora.logger import init_logger, logger
from stylo_flora.inference import (BaseAgent, agent_factory, answer_to_mcq,
                                   reason_input, reason_output, repair,
                                   summarize, tag, translate)
from stylo_flora.benchmarks import BaseBenchmark, benchmark_factory
from stylo_flora import Snippet


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Code task evaluation tool.'
                                               'All the datasets are evaluated by default.')
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model to use.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      choices=[
                          'code_translation',
                          'code_repair',
                          'code2tag',
                          'des_code2tag',
                          'code_summarization',
                          'input_reasoning',
                          'output_reasoning',
                          'mcq_answering',
                      ],
                      help='Specify the code task to evaluate on.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('--dst-lang', type=str, required=False,
                      help='Specify the destination language. Only used for code translation task.')
  parser.add_argument('--result-dir', type=Path, required=True,
                      help='Directory to save the results.')
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
  if args.num_snippets < 0:
    return snippets

  def is_valid(snippet: Snippet) -> bool:
    if not ensure_correct:
      return True
    return math.isclose(calc_correctness([snippet], lang=args.src_lang), 1.0)

  indices = list(range(len(snippets)))
  if args.random:
    random.seed(args.seed)
    random.shuffle(indices)

  candidates = (i for i in indices if is_valid(snippets[i]))
  picked_indices = list(tqdm(islice(candidates, args.num_snippets),
                             desc='Picking snippets', total=args.num_snippets))
  logger.verbose(f'Picked {len(picked_indices)} snippet indices: {picked_indices}')
  return [snippets[i] for i in picked_indices]


def cut_testcases(snippets: Sequence[Snippet], args: argparse.Namespace) -> None:
  if args.num_tests < 0:
    return
  for snippet in snippets:
    if snippet.args.get('io_testcases') and len(snippet.args['io_testcases']) > args.num_tests:
      snippet.args['io_testcases'] = pd.Series(snippet.args['io_testcases']) \
          .sample(n=args.num_tests, random_state=args.seed).tolist()


def load_data(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  data = {}
  with jsonlines.open(path, mode='r') as reader:
    for row in reader:
      data[row['id']] = row
  return data


def save_data(path: Path, data: dict[str, Any], snippets: Sequence[Snippet], corpus: Sequence[Sequence[Snippet]],
              res_orig: Sequence[Any], res_span: Sequence[Sequence[Any]],
              *, returns_snippets: bool = False) -> None:
  def get_variant_output(variant) -> str | None:
    if not variant:
      return None
    return variant.code if returns_snippets else variant
  for i, snippet in enumerate(snippets):
    data[snippet.id] = {
        'id': snippet.id,
        'variants': [variant.code if variant else None for variant in corpus[i]],
        'output': get_variant_output(res_orig[i]),
        'variant_outputs': [get_variant_output(res) for res in res_span[i]],
    }
  with jsonlines.open(path, mode='w') as writer:
    for row in sorted(data.values(), key=itemgetter('id')):
      writer.write(row)
  logger.info(f'Saved data to {path} with {len(data)} snippets.')


def save_result(path: Path, result: dict[str, Any]) -> None:
  with open(path, mode='w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
  logger.info(f'Saved results to {path}.')


def transform(transformer: BaseTransformer, snippets: Sequence[Snippet], data: dict[str, Any],
              args: argparse.Namespace, *, ensure_correct: bool = True) -> Sequence[Sequence[Snippet | None]]:
  # skip transformed snippets that already exist in the result file
  for snippet in snippets:
    if snippet.id in data:
      snippet.args['transformed'] = True

  corpus = transformer.transform(
      snippets=snippets,
      lang=args.src_lang,
      seed=args.seed,
      ensure_correct=ensure_correct,
  )

  # assign the skipped snippets to the corpus
  for i, snippet in enumerate(snippets):
    if snippet.id not in data:
      continue
    for j, variant_code in enumerate(data[snippet.id]['variants']):
      if not variant_code:
        continue
      corpus[i][j] = snippet.replace(code=variant_code)
  return corpus


def perform(snippets: Sequence[Snippet], corpus: Sequence[Sequence[Snippet]],
            data: dict[str, Any], worker: Callable, *,
            returns_snippets: bool = False) -> tuple[Sequence[Any], Sequence[Sequence[Any]]]:
  # skip translated snippets
  for i, snippet in enumerate(snippets):
    if snippet.id not in data:
      continue
    if data[snippet.id]['output']:
      snippet.args['performed'] = True
    for j, variant in enumerate(corpus[i]):
      if data[snippet.id]['variant_outputs'][j]:
        variant.args['performed'] = True

  res_orig, res_span = worker()

  # assign the skipped outputs to the results
  for i, snippet in enumerate(snippets):
    if snippet.id not in data:
      continue
    if data[snippet.id]['output']:
      res_orig[i] = snippet.replace(code=data[snippet.id]['output']) \
          if returns_snippets else data[snippet.id]['output']
    for j, variant in enumerate(corpus[i]):
      if data[snippet.id]['variant_outputs'][j]:
        res_span[i][j] = snippet.replace(code=data[snippet.id]['variant_outputs'][j]) \
            if returns_snippets else data[snippet.id]['variant_outputs'][j]
  return res_orig, res_span


def evaluate_code_translation(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
) -> None:
  if not args.dst_lang:
    raise ValueError('Destination language must be specified for code translation task.')
  snippets = benchmark.load_for_translation(args.src_lang, args.dst_lang)
  snippets = pick_snippets(snippets, args)
  cut_testcases(snippets, args)

  data = load_data(args.data_path)
  corpus = transform(transformer, snippets, data, args)

  def worker() -> tuple[Sequence[Any], Sequence[Sequence[Any]]]:
    res_orig = translate(agent, snippets, args.src_lang, args.dst_lang)
    res_span = [translate(agent, variants, args.src_lang, args.dst_lang)
                for variants in tqdm(corpus, desc='Translating', total=len(corpus), leave=False)]
    return res_orig, res_span
  res_orig, res_span = perform(snippets, corpus, data, worker, returns_snippets=True)

  save_data(args.data_path, data, snippets, corpus, res_orig, res_span, returns_snippets=True)

  logger.info('Evaluating correctness of code translation on the originals.')
  pass_orig = calc_correctness(res_orig, args.dst_lang)
  logger.info('Evaluating correctness of code translation on the variants.')
  pass_span = [calc_correctness(variants, args.dst_lang)
               for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(corpus[0]), leave=False)]

  result = {
      'pass_orig': pass_orig,
      'pass_span': pass_span,
      'pass_span_avg': np.mean(pass_span),
  }
  save_result(args.result_path, result)


def evaluate_code_repair(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
) -> None:
  snippets = benchmark.load_for_repair(args.src_lang)
  snippets = pick_snippets(snippets, args, ensure_correct=False)
  cut_testcases(snippets, args)

  data = load_data(args.data_path)
  corpus = transform(transformer, snippets, data, args, ensure_correct=False)

  def worker() -> tuple[Sequence[Any], Sequence[Sequence[Any]]]:
    logger.info('Repairing on originals.')
    res_orig = repair(agent, snippets, args.src_lang)
    logger.info('Repairing on variants.')
    res_span = [repair(agent, variants, args.src_lang)
                for variants in tqdm(corpus, desc='Repairing', total=len(corpus), leave=False)]
    return res_orig, res_span
  res_orig, res_span = perform(snippets, corpus, data, worker, returns_snippets=True)

  save_data(args.data_path, data, snippets, corpus, res_orig, res_span, returns_snippets=True)

  logger.info('Evaluating correctness of repair on the originals.')
  pass_orig = calc_correctness(res_orig, args.src_lang)
  logger.info('Evaluating correctness of repair on the variants.')
  pass_span = [calc_correctness(variants, args.src_lang)
               for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(corpus[0]), leave=False)]

  result = {
      'pass_orig': pass_orig,
      'pass_span': pass_span,
      'pass_span_avg': np.mean(pass_span),
  }
  save_result(args.result_path, result)


def _evaluate_tagging(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
    *,
    with_desc: bool,
) -> None:
  snippets = benchmark.load_for_tagging(args.src_lang)
  snippets = pick_snippets(snippets, args, ensure_correct=False)

  data = load_data(args.data_path)
  corpus = transform(transformer, snippets, data, args, ensure_correct=False)

  def worker() -> tuple[Sequence[Any], Sequence[Sequence[Any]]]:
    logger.info('Tagging on originals.')
    tags_orig = tag(agent, snippets, args.src_lang, with_desc)
    logger.info('Tagging on variants.')
    tags_span = [tag(agent, variants, args.src_lang, with_desc)
                for variants in tqdm(corpus, desc='Tagging', total=len(corpus), leave=False)]
    return tags_orig, tags_span
  tags_orig, tags_span = perform(snippets, corpus, data, worker)
  gloden_tags = [snippet.args['tags'] for snippet in snippets]

  save_data(args.data_path, data, snippets, corpus, tags_orig, tags_span)

  logger.info('Calculating F1 score of tag classification on the originals.')
  f1_orig = calc_macro_f1(tags_orig, gloden_tags)
  logger.info('Calculating F1 score of tag classification on the variants.')
  f1_span = [calc_macro_f1(variants, gloden_tags)
             for variants in tqdm(zip(*tags_span), desc='Evaluating', total=len(corpus[0]), leave=False)]

  result = {
      'f1_orig': f1_orig,
      'f1_span': f1_span,
      'f1_span_avg': np.mean(f1_span),
  }
  save_result(args.result_path, result)


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


def evaluate_code_summarization(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
) -> None:
  snippets = benchmark.load_for_summarization(args.src_lang)
  snippets = pick_snippets(snippets, args, ensure_correct=False)

  data = load_data(args.data_path)
  corpus = transform(transformer, snippets, data, args, ensure_correct=False)

  def worker() -> tuple[Sequence[Any], Sequence[Sequence[Any]]]:
    logger.info('Summarizing on originals.')
    res_orig = summarize(agent, snippets, args.src_lang)
    logger.info('Summarizing on variants.')
    res_span = [summarize(agent, variants, args.src_lang)
                for variants in tqdm(corpus, desc='Summarizing', total=len(corpus), leave=False)]
    return res_orig, res_span
  res_orig, res_span = perform(snippets, corpus, data, worker)
  human_summaries = [snippet.args['human_summarization'] for snippet in snippets]

  save_data(args.data_path, data, snippets, corpus, res_orig, res_span)

  logger.info('Calculating metrics of code summarization on the originals.')
  bleu_orig = calc_bleu(res_orig, human_summaries)
  meteor_orig = calc_meteor(res_orig, human_summaries)
  rouge_orig = calc_rouge(res_orig, human_summaries)['rougeL']
  bertscore_orig = np.mean(calc_bertscore(res_orig, human_summaries)['f1'])
  overall_orig = np.mean([bleu_orig, meteor_orig, rouge_orig, bertscore_orig])
  logger.info('Calculating metrics of code summarization on the variants.')
  bleu_span = [calc_bleu(variants, human_summaries)
               for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(corpus[0]), leave=False)]
  meteor_span = [calc_meteor(variants, human_summaries)
                 for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(corpus[0]), leave=False)]
  rouge_span = [calc_rouge(variants, human_summaries)['rougeL']
                for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(corpus[0]), leave=False)]
  bertscore_span = [np.mean(calc_bertscore(variants, human_summaries)['f1'])
                    for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(corpus[0]), leave=False)]
  overall_span = np.mean([bleu_span, meteor_span, rouge_span, bertscore_span], axis=0)

  result = {
      'bleu_orig': bleu_orig,
      'bleu_span': bleu_span,
      'bleu_span_avg': np.mean(bleu_span),
      'meteor_orig': meteor_orig,
      'meteor_span': meteor_span,
      'meteor_span_avg': np.mean(meteor_span),
      'rouge_orig': rouge_orig,
      'rouge_span': rouge_span,
      'rouge_span_avg': np.mean(rouge_span),
      'bertscore_orig': bertscore_orig,
      'bertscore_span': bertscore_span,
      'bertscore_span_avg': np.mean(bertscore_span),
      'overall_orig': overall_orig,
      'overall_span': overall_span,
      'overall_span_avg': np.mean(overall_span),
  }
  save_result(args.result_path, result)


def _evaluate_io_reasoning(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
    reason_func: Callable[[BaseAgent, Sequence[Snippet], str], Sequence[Sequence[Snippet]]],
) -> None:
  snippets = benchmark.load_for_io_reasoning(args.src_lang)
  snippets = pick_snippets(snippets, args, ensure_correct=False)

  data = load_data(args.data_path)
  corpus = transform(transformer, snippets, data, args, ensure_correct=False)

  def worker() -> tuple[Sequence[Any], Sequence[Sequence[Any]]]:
    res_orig = reason_func(agent, snippets, args.src_lang)
    res_span = [reason_func(agent, variants, args.src_lang)
                for variants in tqdm(corpus, desc='Reasoning', total=len(corpus), leave=False)]
    return res_orig, res_span
  res_orig, res_span = perform(snippets, corpus, data, worker, returns_snippets=True)

  save_data(args.data_path, data, snippets, corpus, res_orig, res_span, returns_snippets=True)

  logger.info('Calculating metrics of code reasoning on the originals.')
  pass_orig = calc_correctness(res_orig, args.src_lang)
  logger.info('Calculating metrics of code reasoning on the variants.')
  pass_span = [calc_correctness(variants, args.src_lang)
               for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(corpus[0]), leave=False)]

  result = {
      'pass_orig': pass_orig,
      'pass_span': pass_span,
      'pass_span_avg': np.mean(pass_span),
  }
  save_result(args.result_path, result)


def evaluate_input_reasoning(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
) -> None:
  _evaluate_io_reasoning(benchmark, transformer, agent, args, reason_input)


def evaluate_output_reasoning(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
) -> None:
  _evaluate_io_reasoning(benchmark, transformer, agent, args, reason_output)


def evaluate_mcq_answering(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: argparse.Namespace,
) -> None:
  snippets = benchmark.load_for_mcq_answering(args.src_lang)
  snippets = pick_snippets(snippets, args, ensure_correct=False)

  data = load_data(args.data_path)
  corpus = transform(transformer, snippets, data, args, ensure_correct=False)

  def worker() -> tuple[Sequence[Any], Sequence[Sequence[Any]]]:
    logger.info('Answering on originals.')
    res_orig = answer_to_mcq(agent, snippets, args.src_lang)
    logger.info('Answering on variants.')
    res_span = [answer_to_mcq(agent, variants, args.src_lang)
                for variants in tqdm(corpus, desc='Answering', total=len(corpus), leave=False)]
    return res_orig, res_span
  res_orig, res_span = perform(snippets, corpus, data, worker)
  answers = np.array([snippet.args['answer'] for snippet in snippets])

  save_data(args.data_path, data, snippets, corpus, res_orig, res_span)

  acc_orig = np.mean(np.array(res_orig) == answers)
  acc_span = [np.mean(np.array(variants) == answers)
              for variants in zip(*res_span)]

  result = {
      'acc_orig': acc_orig,
      'acc_span': acc_span,
      'acc_span_avg': np.mean(acc_span),
  }
  save_result(args.result_path, result)


def main():
  args = parse_args()
  init_logger(verbose=args.verbose, debug=args.debug)
  benchmark = benchmark_factory(args.dataset)
  transformer = transformer_factory()
  agent = agent_factory(args.model)

  os.makedirs(args.result_dir, exist_ok=True)
  identifier = f'{args.dataset.lower()}_{args.task}_{args.src_lang}_' \
      f'{"to_" + args.dst_lang if args.task == "code_translation" else ""}' \
      f'with_{args.model.replace("/", "-")}_seed{args.seed}'
  args.data_path = args.result_dir / f'data_{identifier}.jsonl'
  args.result_path = args.result_dir / f'result_{identifier}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

  logger.info(f'Evaluating {args.model} on {args.task} task in {args.dataset} '
              f'with working directory {args.result_dir}...')

  evaluator = globals().get(f'evaluate_{args.task}')
  if not evaluator:
    raise ValueError(f'Unsupported task {args.task} for evaluation.')
  evaluator(benchmark, transformer, agent, args)


if __name__ == '__main__':
  main()
