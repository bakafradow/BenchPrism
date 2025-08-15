"""
Assesses the robustness of code task models by the following steps:

1. Extracts source code from different datasets into unified data structure.
2. Applies transformations to the source code to generate a set of transformed code with a code style transformer.
3. Performs code tasks with snippets with the evaluated model.
4. Evaluates the space span by the translated code relative to the original source code.
"""

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
import json
import math
import os
import random
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import islice
from operator import itemgetter
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> Namespace:
  parser = ArgumentParser(description='Code task evaluation tool.'
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


def _pick_snippets(
    snippets: Sequence[Snippet],
    args: Namespace,
    *,
    ensure_correct: bool = True,
) -> Sequence[Snippet]:
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


def _cut_testcases(
    snippets: Sequence[Snippet],
    args: Namespace,
) -> None:
  if snippets and not snippets[0].args.get('io_testcases'):
    return
  if args.num_tests < 0:
    return
  for snippet in snippets:
    if snippet.args.get('io_testcases') and len(snippet.args['io_testcases']) > args.num_tests:
      snippet.args['io_testcases'] = pd.Series(snippet.args['io_testcases']) \
          .sample(n=args.num_tests, random_state=args.seed).tolist()


def _load_data(
    path: Path,
) -> dict[str, Any]:
  if not path.exists():
    return {}
  data = {}
  with jsonlines.open(path, mode='r') as reader:
    for row in reader:
      data[row['id']] = row
  return data


def _save_data(
    path: Path,
    data: dict[str, Any],
    snippets: Sequence[Snippet],
    corpus: Sequence[Sequence[Snippet]],
    res_orig: Sequence[Any] | None = None,
    res_span: Sequence[Sequence[Any]] | None = None,
    *,
    returns_snippets: bool = False,
) -> None:
  def get_variant_output(variant) -> str | None:
    if not variant:
      return None
    return variant.code if returns_snippets else variant
  for i, snippet in enumerate(snippets):
    data.setdefault(snippet.id, {})
    data[snippet.id].update({
        'id': snippet.id,
        'variants': [variant.code if variant else None for variant in corpus[i]],
    })
    if res_orig:
      data[snippet.id]['output'] = get_variant_output(res_orig[i])
    if res_span:
      data[snippet.id]['variant_outputs'] = [get_variant_output(variant) for variant in res_span[i]]

  with jsonlines.open(path, mode='w') as writer:
    for row in sorted(data.values(), key=itemgetter('id')):
      writer.write(row)
  logger.info(f'Saved data to {path} with {len(data)} rows.')


def _save_result(
    path: Path,
    result: dict[str, Any],
) -> None:
  with open(path, mode='w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
  logger.info(f'Saved results to {path}.')


def _transform_with(
    transformer: BaseTransformer,
    snippets: Sequence[Snippet],
    data: dict[str, Any],
    args: Namespace,
    *,
    ensure_correct: bool = True,
) -> Sequence[Sequence[Snippet | None]]:
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


def _perform_with(
    worker: Callable,
    snippets: Sequence[Snippet],
    corpus: Sequence[Sequence[Snippet]],
    data: dict[str, Any],
    *,
    returns_snippets: bool = False,
) -> tuple[Sequence[Any], Sequence[Sequence[Any]]]:
  # skip snippets that already have outputs in data
  for i, snippet in enumerate(snippets):
    if snippet.id not in data:
      continue
    if data[snippet.id].get('output'):
      snippet.args['performed'] = True
    if not data[snippet.id].get('variant_outputs'):
      continue
    for j, variant in enumerate(corpus[i]):
      if data[snippet.id]['variant_outputs'][j]:
        variant.args['performed'] = True

  res_orig, res_span = worker()

  # assign outputs in data to the responses
  for i, snippet in enumerate(snippets):
    if snippet.id not in data:
      continue
    if data[snippet.id].get('output'):
      res_orig[i] = snippet.replace(code=data[snippet.id]['output']) \
          if returns_snippets else data[snippet.id]['output']
    if not data[snippet.id].get('variant_outputs'):
      continue
    for j, variant in enumerate(corpus[i]):
      if data[snippet.id]['variant_outputs'][j]:
        res_span[i][j] = snippet.replace(code=data[snippet.id]['variant_outputs'][j]) \
            if returns_snippets else data[snippet.id]['variant_outputs'][j]
  return res_orig, res_span


def _evaluate_task_template(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
    load_snippets_func: Callable[[BaseBenchmark, Namespace], Sequence[Snippet]],
    perform_task_func: Callable[[BaseAgent, Sequence[Snippet], str], Sequence[Any]],
    evaluate_metrics_func: Callable[[Sequence[Snippet], Sequence[Sequence[Snippet]],
                                     Sequence[Any], Sequence[Sequence[Any]],
                                     Namespace], dict[str, Any]],
    *,
    ensure_correct: bool = True,
) -> None:
  logger.info(f'Loading snippets for {args.task} from {args.dataset}...')
  snippets = load_snippets_func(benchmark, args)
  snippets = _pick_snippets(snippets, args, ensure_correct=ensure_correct)
  _cut_testcases(snippets, args)

  data = _load_data(args.data_path)
  logger.info('Transforming styles of the code snippets...')
  corpus = _transform_with(transformer, snippets, data, args, ensure_correct=ensure_correct)
  _save_data(args.data_path, data, snippets, corpus,
             returns_snippets=args.returns_snippets)
  num_styles = len(corpus[0]) if corpus else 0

  def worker() -> tuple[Sequence[Any], Sequence[Sequence[Any]]]:
    res_orig = perform_task_func(agent, snippets, args)
    res_span = [perform_task_func(agent, variants, args)
                for variants in tqdm(corpus, desc=args.task.capitalize(),
                                     total=len(corpus), leave=False)]
    return res_orig, res_span
  logger.info(f'Performing {args.task} with {args.model}...')
  res_orig, res_span = _perform_with(worker, snippets, corpus, data, returns_snippets=args.returns_snippets)
  _save_data(args.data_path, data, snippets, corpus,
             res_orig, res_span, returns_snippets=args.returns_snippets)

  # eliminate `None`s by filtering
  indices_to_eval = [i for i, res in enumerate(res_orig) if res]
  snippets_to_eval = [snippets[i] for i in indices_to_eval]
  corpus_to_eval = [corpus[i] for i in indices_to_eval]
  res_orig_to_eval = [res_orig[i] for i in indices_to_eval]
  res_span_to_eval = [res_span[i] for i in indices_to_eval]
  # eliminate `None`s by padding
  fallbacks = [variants.count(None) for variants in zip(*corpus_to_eval)]
  corpus_to_eval = [[variant or snippets_to_eval[i] for variant in variants]
                    for i, variants in enumerate(corpus_to_eval)]
  res_span_to_eval = [[res or res_orig_to_eval[i] for res in responses]
                      for i, responses in enumerate(res_span_to_eval)]
  assert all(snippets_to_eval)
  assert all(variant for variants in corpus_to_eval for variant in variants)
  assert all(res_orig_to_eval)
  assert all(res for responses in res_span_to_eval for res in responses)

  logger.info('Calculating metrics for the responses...')
  result = evaluate_metrics_func(snippets_to_eval, corpus_to_eval, res_orig_to_eval, res_span_to_eval, args)
  codebleu = [calc_codebleu([snippet.code for snippet in snippets_to_eval],
                            [variant.code for variant in variants], args.src_lang)['codebleu']
              for variants in tqdm(zip(*corpus_to_eval), desc='Calculating CodeBLEU',
                                   total=num_styles, leave=False)]
  result.update({
      'num_original': len(snippets),
      'num_valid': len(snippets_to_eval),
      'num_skipped': len(snippets) - len(indices_to_eval),
      'num_styles': num_styles,
      'fallbacks': fallbacks,
      'fallback_rate': np.mean(fallbacks) / num_styles,
      'codebleu': codebleu,
      'codebleu_avg': np.mean(codebleu),
  })
  _save_result(args.result_path, result)


def evaluate_code_translation(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
) -> None:
  if not args.dst_lang:
    raise ValueError('Destination language must be specified for code translation task.')

  def evaluate_metrics(
          snippets: Sequence[Snippet], corpus: Sequence[Sequence[Snippet]],
          res_orig: Sequence[Any], res_span: Sequence[Sequence[Any]],
          args: Namespace) -> dict[str, Any]:
    pass_orig = calc_correctness(res_orig, args.dst_lang)
    pass_span = [calc_correctness(variants, args.dst_lang)
                 for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    return {
        'pass_orig': pass_orig,
        'pass_span': pass_span,
        'pass_span_avg': np.mean(pass_span),
    }

  _evaluate_task_template(
      benchmark, transformer, agent, args,
      load_snippets_func=lambda b, a: b.load_for_translation(a.src_lang, a.dst_lang),
      perform_task_func=lambda ag, sn, a: translate(ag, sn, a.src_lang, args.dst_lang),
      evaluate_metrics_func=evaluate_metrics,
      ensure_correct=True,
  )


def evaluate_code_repair(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
) -> None:
  def evaluate_metrics(
          snippets: Sequence[Snippet], corpus: Sequence[Sequence[Snippet]],
          res_orig: Sequence[Any], res_span: Sequence[Sequence[Any]],
          args: Namespace) -> dict[str, Any]:
    pass_orig = calc_correctness(res_orig, args.src_lang)
    pass_span = [calc_correctness(variants, args.src_lang)
                 for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    return {
        'pass_orig': pass_orig,
        'pass_span': pass_span,
        'pass_span_avg': np.mean(pass_span),
    }

  _evaluate_task_template(
      benchmark, transformer, agent, args,
      load_snippets_func=lambda b, a: b.load_for_repair(a.src_lang),
      perform_task_func=lambda ag, sn, a: repair(ag, sn, a.src_lang),
      evaluate_metrics_func=evaluate_metrics,
      ensure_correct=False,
  )


def _evaluate_tag_classification(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
    *,
    with_desc: bool,
) -> None:
  def evaluate_metrics(
          snippets: Sequence[Snippet], corpus: Sequence[Sequence[Snippet]],
          res_orig: Sequence[Any], res_span: Sequence[Sequence[Any]],
          args: Namespace) -> dict[str, Any]:
    gloden_tags = [snippet.args['tags'] for snippet in snippets]
    f1_orig = calc_macro_f1(res_orig, gloden_tags)
    f1_span = [calc_macro_f1(variants, gloden_tags)
               for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    return {
        'f1_orig': f1_orig,
        'f1_span': f1_span,
        'f1_span_avg': np.mean(f1_span),
    }

  _evaluate_task_template(
      benchmark, transformer, agent, args,
      load_snippets_func=lambda b, a: b.load_for_tagging(a.src_lang),
      perform_task_func=lambda ag, sn, a: tag(ag, sn, a.src_lang, with_desc),
      evaluate_metrics_func=evaluate_metrics,
      ensure_correct=False,
  )


def evaluate_code2tag(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
) -> None:
  _evaluate_tag_classification(benchmark, transformer, agent, args, with_desc=False)


def evaluate_des_code2tag(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
) -> None:
  _evaluate_tag_classification(benchmark, transformer, agent, args, with_desc=True)


def evaluate_code_summarization(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
) -> None:
  def evaluate_metrics(
          snippets: Sequence[Snippet], corpus: Sequence[Sequence[Snippet]],
          res_orig: Sequence[Any], res_span: Sequence[Sequence[Any]],
          args: Namespace) -> dict[str, Any]:
    human_summaries = [snippet.args['human_summarization'] for snippet in snippets]
    bleu_orig = calc_bleu(res_orig, human_summaries)
    meteor_orig = calc_meteor(res_orig, human_summaries)
    rouge_orig = calc_rouge(res_orig, human_summaries)['rougeL']
    bertscore_orig = np.mean(calc_bertscore(res_orig, human_summaries)['f1'])
    overall_orig = np.mean([bleu_orig, meteor_orig, rouge_orig, bertscore_orig])
    bleu_span = [calc_bleu(variants, human_summaries)
                 for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    meteor_span = [calc_meteor(variants, human_summaries)
                   for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    rouge_span = [calc_rouge(variants, human_summaries)['rougeL']
                  for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    bertscore_span = [np.mean(calc_bertscore(variants, human_summaries)['f1'])
                      for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    overall_span = np.mean([bleu_span, meteor_span, rouge_span, bertscore_span], axis=0)
    return {
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

  _evaluate_task_template(
      benchmark, transformer, agent, args,
      load_snippets_func=lambda b, a: b.load_for_summarization(a.src_lang),
      perform_task_func=lambda ag, sn, a: summarize(ag, sn, a.src_lang),
      evaluate_metrics_func=evaluate_metrics,
      ensure_correct=False,
  )


def _evaluate_io_reasoning(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
    reason_func: Callable[[BaseAgent, Sequence[Snippet], str], Sequence[Sequence[Any]]],
) -> None:
  def evaluate_metrics(
          snippets: Sequence[Snippet], corpus: Sequence[Sequence[Snippet]],
          res_orig: Sequence[Any], res_span: Sequence[Sequence[Any]],
          args: Namespace) -> dict[str, Any]:
    pass_orig = calc_correctness(res_orig, args.src_lang)
    pass_span = [calc_correctness(variants, args.src_lang)
                 for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    return {
        'pass_orig': pass_orig,
        'pass_span': pass_span,
        'pass_span_avg': np.mean(pass_span),
    }

  _evaluate_task_template(
      benchmark, transformer, agent, args,
      load_snippets_func=lambda b, a: b.load_for_io_reasoning(a.src_lang),
      perform_task_func=lambda ag, sn, a: reason_func(ag, sn, a.src_lang),
      evaluate_metrics_func=evaluate_metrics,
      ensure_correct=True,
  )


def evaluate_input_reasoning(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
) -> None:
  _evaluate_io_reasoning(benchmark, transformer, agent, args, reason_input)


def evaluate_output_reasoning(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
) -> None:
  _evaluate_io_reasoning(benchmark, transformer, agent, args, reason_output)


def evaluate_mcq_answering(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    args: Namespace,
) -> None:
  def evaluate_metrics(
          snippets: Sequence[Snippet], corpus: Sequence[Sequence[Snippet]],
          res_orig: Sequence[Any], res_span: Sequence[Sequence[Any]],
          args: Namespace) -> dict[str, Any]:
    answers = np.array([snippet.args['answer'] for snippet in snippets])
    acc_orig = np.mean(np.array(res_orig) == answers)
    acc_span = [np.mean(np.array(variants) == answers)
                for variants in zip(*res_span)]
    return {
        'acc_orig': acc_orig,
        'acc_span': acc_span,
        'acc_span_avg': np.mean(acc_span),
    }

  _evaluate_task_template(
      benchmark, transformer, agent, args,
      load_snippets_func=lambda b, a: b.load_for_mcq_answering(a.src_lang),
      perform_task_func=lambda ag, sn, a: answer_to_mcq(ag, sn, a.src_lang),
      evaluate_metrics_func=evaluate_metrics,
      ensure_correct=False,
  )


def main():
  args = parse_args()
  init_logger(verbose=args.verbose, debug=args.debug)
  logger.info(f'Initializing benchmark {args.dataset}...')
  benchmark = benchmark_factory(args.dataset)
  logger.info('Initializing transformer...')
  transformer = transformer_factory()
  logger.info(f'Initializing model {args.model}...')
  agent = agent_factory(args.model)

  os.makedirs(args.result_dir, exist_ok=True)
  identifier = f'{args.dataset.lower()}_{args.task}_{args.src_lang}_' \
      f'{"to_" + args.dst_lang if args.task == "code_translation" else ""}' \
      f'with_{args.model.replace("/", "-")}_seed{args.seed}'
  args.data_path = args.result_dir / f'data_{identifier}.jsonl'
  args.result_path = args.result_dir / f'result_{identifier}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
  args.returns_snippets = args.task in [
      'code_translation', 'code_repair', 'input_reasoning', 'output_reasoning',
  ]

  evaluator = globals().get(f'evaluate_{args.task}')
  if not evaluator:
    raise ValueError(f'Unsupported task {args.task} for evaluation.')
  evaluator(benchmark, transformer, agent, args)


if __name__ == '__main__':
  main()
