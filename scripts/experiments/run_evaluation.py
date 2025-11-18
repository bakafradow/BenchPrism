"""
Assesses the robustness of code task models by the following steps:

1. Extracts source code from different datasets into unified data structure.
2. Applies transformations to the source code to generate a set of transformed code with a code style transformer.
3. Performs code tasks with snippets with the evaluated model.
4. Evaluates the space span by the translated code relative to the original source code.
"""

import json
import math
import os
import random
import re
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from collections.abc import MutableSequence as MSeq
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import islice, repeat
from operator import itemgetter
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm

from stylo_flora import IOTestCase, Snippet, setting_dict
from stylo_flora.benchmarks import BaseBenchmark, benchmark_factory
from stylo_flora.inference import (BaseAgent, BaseTask, task_factory,
                                   agent_factory, task_worker)
from stylo_flora.logger import init_logger, logger
from stylo_flora.metrics import (calc_bertscore, calc_bleu, calc_codebleu,
                                 calc_correctness, calc_coverage,
                                 calc_macro_f1, calc_meteor, calc_rouge)
from stylo_flora.transformer.base import BaseTransformer, transformer_factory

_MetricsEvaluator = Callable[[Seq[str], Seq[Seq[str]], Seq[Snippet], Namespace],
                            dict[str, Any]]


def parse_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model to use.')
  parser.add_argument('--model-path', type=str, required=False,
                      help='Specify the local/HF path to load model. Only used for open-source models.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      choices=[
                          'code_translation',
                          'code_repair',
                          'code2tag',
                          'descode2tag',
                          'code_summarization',
                          'input_reasoning',
                          'output_reasoning',
                          'mcq_answering',
                          'test_generation',
                      ],
                      help='Specify the code task to evaluate on.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('--dst-lang', type=str, required=False,
                      help='Specify the destination language. Only used for code translation task.')
  parser.add_argument('--result-dir', type=Path, required=True,
                      help='Directory to save the results.')
  parser.add_argument('--log-path', type=Path, required=False,
                      help='Path to save the log file. If not set, only logs to console.')
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
  parser.add_argument('--transform-only', action='store_true', default=False,
                      help='If set, terminates after code transformation WITHOUT inference.')
  parser.add_argument('--batch-api', action='store_true', default=False,
                      help='If set, uses batch API and terminates WITHOUT evaluation. Only for proprietary models.')
  parser.add_argument('--evaluate-only', action='store_true', default=False,
                      help='If set, only calculates metrics with existing data without transformation and inference.')
  args = parser.parse_args()
  return args


def _pick_snippets(
    snippets: Seq[Snippet],
    args: Namespace,
    *,
    ensure_correct: bool = True,
) -> Seq[Snippet]:
  if args.num_snippets < 0:
    args.num_snippets = len(snippets)

  def is_valid(snippet: Snippet) -> bool:
    if not ensure_correct:
      return True
    return math.isclose(calc_correctness([snippet.data['code']], [snippet.data], lang=args.src_lang), 1.0)

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
    snippets: Seq[Snippet],
    args: Namespace,
) -> None:
  if snippets and not snippets[0].data.get('io_testcases'):
    return
  if args.num_tests < 0:
    return
  for snippet in snippets:
    if snippet.data.get('io_testcases') and len(snippet.data['io_testcases']) > args.num_tests:
      snippet.data['io_testcases'] = pd.Series(snippet.data['io_testcases']) \
          .sample(n=args.num_tests, random_state=args.seed).tolist()


def _load_jsonl(
    path: Path,
) -> dict:
  if not path.exists():
    return {}
  with jsonlines.open(path, mode='r') as reader:
    data = {row['id']: row for row in reader}
    return data


def _load_outputs(
    path: Path,
    snippets: Seq[Snippet],
) -> tuple[MSeq[Any], MSeq[MSeq[Any]]]:
  output_data = _load_jsonl(path)

  res_orig: MSeq[Any] = []
  res_span: MSeq[MSeq[Any]] = []
  for snippet in snippets:
    res_orig.append(output_data.get(snippet.id, {}).get('output', None))
    res_span.append(output_data.get(snippet.id, {}).get('variant_outputs', []))
  return res_orig, res_span


def _save_json(
    path: Path,
    obj: dict,
) -> None:
  with open(path, mode='w') as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_variants(
    path: Path,
    data: dict[str, Any],
    snippets: Seq[Snippet],
    corpus: Seq[Seq[str | None]],
) -> None:
  for i, snippet in enumerate(snippets):
    data.setdefault(snippet.id, {})
    data[snippet.id].update({
        'id': snippet.id,
        'variants': corpus[i],
    })

  with jsonlines.open(path, mode='w') as writer:
    for row in sorted(data.values(), key=itemgetter('id')):
      writer.write(row)
  logger.info(f'Saved variants to {path} with {len(data)} rows.')


def _save_outputs(
    path: Path,
    data: dict[str, Any],
    snippets: Seq[Snippet],
    res_orig: Seq[Any],
    res_span: Seq[Seq[Any]],
) -> None:
  for i, snippet in enumerate(snippets):
    data.setdefault(snippet.id, {})
    data[snippet.id].setdefault('id', snippet.id)
    if res_orig:
      data[snippet.id]['output'] = res_orig[i]
    if res_span:
      data[snippet.id]['variant_outputs'] = res_span[i]

  with jsonlines.open(path, mode='w') as writer:
    for row in sorted(data.values(), key=itemgetter('id')):
      writer.write(row)
  logger.info(f'Saved outputs to {path} with {len(data)} rows.')


def _transform_with(
    transformer: BaseTransformer,
    snippets: Seq[Snippet],
    args: Namespace,
    *,
    check: bool = True,
) -> list[list[str | None]]:
  variant_data = _load_jsonl(args.variants_path)

  start_time = time.perf_counter()
  corpus: list[list[str | None]] = []
  for snippet in tqdm(snippets, desc='Transforming styles', total=len(snippets), leave=False):
    cached_variants = variant_data.get(snippet.id, {}).get('variants', [])
    seqs_to_skip = {i for i, code in enumerate(cached_variants) if code}

    variants = transformer.transform(snippet, check=check, seqs_to_skip=seqs_to_skip)

    if len(cached_variants) == len(variants):
      for i in range(len(variants)):
        variants[i] = variants[i] or cached_variants[i]

    corpus.append(variants)
  transform_time = time.perf_counter() - start_time
  logger.info(f'Spanned coding styles in {transform_time:.2f}s.')

  _save_variants(args.variants_path, variant_data, snippets, corpus)
  return corpus


def _perform_with(
    agent: BaseAgent,
    task: BaseTask,
    snippets: Seq[Snippet],
    corpus: Seq[Seq[str | None]],
    args: Namespace,
) -> tuple[MSeq[Any], MSeq[MSeq[Any]]]:
  output_data = _load_jsonl(args.outputs_path)

  start_time = time.perf_counter()
  res_orig: MSeq[Any] = []
  res_span: MSeq[MSeq[Any]] = []
  for i, snippet in tqdm(enumerate(snippets), desc=f'Performing {task.__class__.__name__}', total=len(snippets), leave=False):
    cached_output = output_data.get(snippet.id, {}).get('output', None)
    output_orig = cached_output or task_worker(agent, task, snippet)

    cached_variant_outputs = output_data.get(snippet.id, {}).get('variant_outputs', [])
    seqs_to_skip = {j for j, output in enumerate(cached_variant_outputs) if output}
    var_snippets = [None if j in seqs_to_skip else snippet.replace(code=code)
                    for j, code in enumerate(corpus[i])]

    with ThreadPoolExecutor(max_workers=setting_dict['agent']['max_workers']) as executor:
      output_span = list(tqdm(executor.map(task_worker, repeat(agent), repeat(task), var_snippets),
                              desc='Inferencing', total=len(var_snippets), leave=False))

    if len(cached_variant_outputs) == len(output_span):
      for j in range(len(output_span)):
        output_span[j] = output_span[j] or cached_variant_outputs[j]
    
    res_orig.append(output_orig)
    res_span.append(output_span)
  inference_time = time.perf_counter() - start_time
  logger.info(f'Finish inference in {inference_time:.2f}s.')

  _save_outputs(args.outputs_path, output_data, snippets, res_orig, res_span)
  return res_orig, res_span


def _batch_with(
    agent: BaseAgent,
    task: BaseTask,
    snippets: Seq[Snippet],
    corpus: Seq[Seq[str | None]],
    args: Namespace,
) -> None:
  output_data = _load_jsonl(args.outputs_path)

  requests = []
  for i, snippet in enumerate(snippets):
    cached_output = output_data.get(snippet.id, {}).get('output', None)
    if not cached_output:
      sys_prompt, user_prompt = task.get_prompt(snippet)
      req = agent.create_batch_request(f'{snippet.id}_orig', sys_prompt, user_prompt)
      requests.append(req)

    cached_variant_outputs = output_data.get(snippet.id, {}).get('variant_outputs', [])
    seqs_to_skip = {j for j, output in enumerate(cached_variant_outputs) if output}

    for j, code in enumerate(corpus[i]):
      if j in seqs_to_skip:
        continue
      var_snippet = snippet.replace(code=code)
      sys_prompt, user_prompt = task.get_prompt(var_snippet)
      req = agent.create_batch_request(f'{snippet.id}_{j}', sys_prompt, user_prompt)
      requests.append(req)
  agent.submit_batch_job(requests)

  # save dummy outputs
  res_orig = [None] * len(snippets)
  res_span = [[None] * len(variants) for variants in corpus]
  _save_outputs(args.outputs_path, output_data, snippets, res_orig, res_span)


def _evaluate_task_template(
    snippets: Seq[Snippet],
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
    *,
    metrics_evaluator: _MetricsEvaluator,
    check: bool = True,
) -> None:
  _cut_testcases(snippets, args)
  snippets = _pick_snippets(snippets, args, ensure_correct=check)

  logger.info(f'Transforming styles of {len(snippets)} code snippets...')
  corpus = _transform_with(transformer, snippets, args, check=check)
  num_styles = len(corpus[0]) if corpus else 0
  if args.evaluate_only:
    logger.info(f'--evaluate-only is set, only evaluating outputs from {args.outputs_path}...')
    res_orig, res_span = _load_outputs(args.outputs_path, snippets)
  else:
    if args.transform_only:
      logger.info('--transform-only is set, skipping inference.')
      return
    if args.batch_api:
      logger.info('Submitting code tasks via batch API...')
      _batch_with(agent, task, snippets, corpus, args)
      logger.info('--batch-api is set, skipping evaluation.')
      return
    logger.info(f'Performing {args.task} with {args.model}...')
    res_orig, res_span = _perform_with(agent, task, snippets, corpus, args)

  # eliminate `None`s by filtering
  indices_filtered = [i for i, res in enumerate(res_orig) if res]
  snippets_filtered = [snippets[i] for i in indices_filtered]
  corpus_filtered = [corpus[i] for i in indices_filtered]
  res_orig_filtered = [res_orig[i] for i in indices_filtered]
  res_span_filtered = [res_span[i] for i in indices_filtered]
  # eliminate `None`s by padding
  fallbacks = [variants.count(None) for variants in zip(*corpus_filtered)]
  corpus_to_eval = [[variant or snippets_filtered[i].data['code'] for variant in variants]
                    for i, variants in enumerate(corpus_filtered)]
  res_span_to_eval = [[res or res_orig_filtered[i] for res in responses]
                      for i, responses in enumerate(res_span_filtered)]
  assert all(snippets_filtered)
  assert all(variant for variants in corpus_to_eval for variant in variants)
  assert all(res_orig_filtered)
  assert all(res for responses in res_span_to_eval for res in responses)

  logger.info('Calculating metrics for the responses...')
  result = metrics_evaluator(res_orig_filtered, res_span_to_eval, snippets_filtered, args)
  codebleu = [calc_codebleu([snippet.data['code'] for snippet in snippets_filtered],
                            variants, args.src_lang)['codebleu']
              for variants in tqdm(zip(*corpus_to_eval), desc='Calculating CodeBLEU',
                                   total=num_styles, leave=False)]
  result.update({
      'num_original': len(snippets),
      'num_valid': len(snippets_filtered),
      'num_skipped': len(snippets) - len(indices_filtered),
      'num_styles': num_styles,
      'fallbacks': fallbacks,
      'fallback_rate': np.mean(fallbacks) / len(indices_filtered),
      'codebleu': codebleu,
      'codebleu_avg': np.mean(codebleu),
      'token_count': agent.token_count,
  })
  _save_json(args.result_path, result)
  logger.info(f'Saved results to {args.result_path}.')


def evaluate_code_translation(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  if not args.dst_lang:
    raise ValueError('Destination language must be specified for code translation task.')

  def evaluate_metrics(
      res_orig: Seq[str], res_span: Seq[Seq[str]],
      snippets: Seq[Snippet], args: Namespace,
  ) -> dict[str, Any]:
    args_list = [snippet.data for snippet in snippets]
    pass_orig = calc_correctness(res_orig, args_list, args.dst_lang)
    pass_span = [calc_correctness(variants, args_list, args.dst_lang)
                 for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    return {
        'pass_orig': pass_orig,
        'pass_span': pass_span,
        'pass_span_avg': np.mean(pass_span),
    }
  
  snippets = benchmark.load_for_translation(args.src_lang, args.dst_lang)
  _evaluate_task_template(
      snippets, transformer, agent, task, args,
      metrics_evaluator=evaluate_metrics, check=True,
  )


def evaluate_code_repair(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  def evaluate_metrics(
      res_orig: Seq[str], res_span: Seq[Seq[str]],
      snippets: Seq[Snippet], args: Namespace,
  ) -> dict[str, Any]:
    args_list = [snippet.data for snippet in snippets]
    pass_orig = calc_correctness(res_orig, args_list, args.src_lang)
    pass_span = [calc_correctness(variants, args_list, args.src_lang)
                 for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    return {
        'pass_orig': pass_orig,
        'pass_span': pass_span,
        'pass_span_avg': np.mean(pass_span),
    }

  snippets = benchmark.load_for_repair(args.src_lang)
  _evaluate_task_template(
      snippets, transformer, agent, task, args,
      metrics_evaluator=evaluate_metrics,
      check=False,
  )


def _evaluate_tag_classification(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  def evaluate_metrics(
      res_orig: Seq[Seq[str]], res_span: Seq[Seq[Seq[str]]],
      snippets: Seq[Snippet], args: Namespace,
  ) -> dict[str, Any]:
    gloden_tags = [snippet.data['tags'] for snippet in snippets]
    f1_orig = calc_macro_f1(res_orig, gloden_tags)
    f1_span = [calc_macro_f1(variants, gloden_tags)
               for variants in tqdm(zip(*res_span), desc='Evaluating', total=len(res_span[0]), leave=False)]
    return {
        'f1_orig': f1_orig,
        'f1_span': f1_span,
        'f1_span_avg': np.mean(f1_span),
    }

  snippets = benchmark.load_for_tagging(args.src_lang)
  _evaluate_task_template(
      snippets, transformer, agent, task, args,
      metrics_evaluator=evaluate_metrics,
      check=False,
  )


def evaluate_code2tag(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  _evaluate_tag_classification(benchmark, transformer, agent, task, args)


def evaluate_descode2tag(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  _evaluate_tag_classification(benchmark, transformer, agent, task, args)


def evaluate_code_summarization(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  def evaluate_metrics(
      res_orig: Seq[str], res_span: Seq[Seq[str]],
      snippets: Seq[Snippet], args: Namespace,
  ) -> dict[str, Any]:
    human_summaries = [snippet.data['human_summarization'] for snippet in snippets]
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

  snippets = benchmark.load_for_summarization(args.src_lang)
  _evaluate_task_template(
      snippets, transformer, agent, task, args,
      metrics_evaluator=evaluate_metrics,
      check=False,
  )


def _evaluate_io_reasoning(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  def evaluate_metrics(
      res_orig: Seq[str], res_span: Seq[Seq[str]],
      snippets: Seq[Snippet], args: Namespace,
  ) -> dict[str, Any]:
    res_code_orig = [snippet.data['code'].replace('????', res)
                     for snippet, res in zip(snippets, res_orig)]
    res_code_span = [[snippet.data['code'].replace('????', res) for res in res_list]
                     for snippet, res_list in zip(snippets, res_span)]
    args_list = [snippet.data for snippet in snippets]
    pass_orig = calc_correctness(res_code_orig, args_list, args.src_lang)
    pass_span = [calc_correctness(variants, args_list, args.src_lang)
                 for variants in tqdm(zip(*res_code_span), desc='Evaluating', total=len(res_code_span[0]), leave=False)]
    return {
        'pass_orig': pass_orig,
        'pass_span': pass_span,
        'pass_span_avg': np.mean(pass_span),
    }

  snippets = benchmark.load_for_io_reasoning(args.src_lang)
  _evaluate_task_template(
      snippets, transformer, agent, task, args,
      metrics_evaluator=evaluate_metrics,
      check=True,
  )


def evaluate_input_reasoning(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  _evaluate_io_reasoning(benchmark, transformer, agent, task, args)


def evaluate_output_reasoning(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  _evaluate_io_reasoning(benchmark, transformer, agent, task, args)


def evaluate_mcq_answering(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  def evaluate_metrics(
      res_orig: Seq[str], res_span: Seq[Seq[str]],
      snippets: Seq[Snippet], args: Namespace,
  ) -> dict[str, Any]:
    answers = np.array([snippet.data['answer'] for snippet in snippets])
    acc_orig = np.mean(np.array(res_orig) == answers)
    acc_span = [np.mean(np.array(variants) == answers)
                for variants in zip(*res_span)]
    return {
        'acc_orig': acc_orig,
        'acc_span': acc_span,
        'acc_span_avg': np.mean(acc_span),
    }

  snippets = benchmark.load_for_mcq_answering(args.src_lang)
  _evaluate_task_template(
      snippets, transformer, agent, task, args,
      metrics_evaluator=evaluate_metrics,
      check=False,
  )


def evaluate_test_generation(
    benchmark: BaseBenchmark,
    transformer: BaseTransformer,
    agent: BaseAgent,
    task: BaseTask,
    args: Namespace,
) -> None:
  def evaluate_metrics(
      res_orig: Seq[Seq[Any]], res_span: Seq[Seq[Seq[Any]]],
      snippets: Seq[Snippet], args: Namespace,
  ) -> dict[str, Any]:
    code_list = [snippet.data['code'] for snippet in snippets]
    tc_list_orig = [[IOTestCase.from_list(l) for l in res]
                    for res in res_orig]
    tc_lists_span = [[[IOTestCase.from_list(l) for l in res]
                      for res in res_list]
                     for res_list in res_span]
    cov_orig = calc_coverage(code_list, tc_list_orig, args.src_lang)
    cov_span = [calc_coverage(code_list, tc_list, args.src_lang)
                for tc_list in tqdm(zip(*tc_lists_span), desc='Evaluating',
                                    total=len(res_span[0]), leave=False)]
    pass_span = [cov['pass_rate'] for cov in cov_span]
    line_cov_span = [cov['line_cov_rate'] for cov in cov_span]
    branch_cov_span = [cov['branch_cov_rate'] for cov in cov_span]
    return {
        'pass_orig': cov_orig['pass_rate'],
        'pass_span': pass_span,
        'pass_span_avg': np.mean(pass_span),
        'line_cov_orig': cov_orig['line_cov_rate'],
        'line_cov_span': line_cov_span,
        'line_cov_span_avg': np.mean(line_cov_span),
        'branch_cov_orig': cov_orig['branch_cov_rate'],
        'branch_cov_span': branch_cov_span,
        'branch_cov_span_avg': np.mean(branch_cov_span),
    }

  snippets = benchmark.load_for_test_generation(args.src_lang)
  _evaluate_task_template(
      snippets, transformer, agent, task, args,
      metrics_evaluator=evaluate_metrics,
      check=True,
  )


def main():
  args = parse_args()
  init_logger(path=args.log_path, verbose=args.verbose, debug=args.debug)
  logger.info(f'Initializing benchmark {args.dataset}...')
  benchmark = benchmark_factory(dataset=args.dataset)
  logger.info('Initializing transformer...')
  transformer = transformer_factory(lang=args.src_lang, seed=args.seed)
  logger.info(f'Initializing model {args.model}...')
  agent = agent_factory(name=args.model, model_path=args.model_path)
  logger.info(f'Initializing task {args.task}...')
  task = task_factory(args.task, **dict(args._get_kwargs()))

  os.makedirs(args.result_dir, exist_ok=True)
  args.variants_path = args.result_dir /\
      f'variants_{args.dataset.lower()}_{args.task}_{args.src_lang}_seed{args.seed}.jsonl'
  args.identifier = f'{args.dataset.lower()}_{args.task}_{args.src_lang}'
  if args.task == 'code_translation':
    args.identifier += f'{"_to_" + args.dst_lang}'
  args.identifier += f'_with_{re.sub(r"[/: ]+", "-", args.model)}_seed{args.seed}'
  args.outputs_path = args.result_dir /\
      f'outputs_{args.identifier}.jsonl'
  args.result_path = args.result_dir /\
      f'results_{args.identifier}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

  evaluator = globals().get(f'evaluate_{args.task}')
  if not evaluator:
    raise ValueError(f'Unsupported task {args.task} for evaluation.')
  evaluator(benchmark, transformer, agent, task, args)


if __name__ == '__main__':
  main()
