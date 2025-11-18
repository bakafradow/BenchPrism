"""
Prepares IPython enviromnment for fine-grained debugging of transformations.

Usage:
  ipython -i <path_to_this_file> -- -d <dataset> -t <task> --src-lang <src-lang>
"""

import ast
import json
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

from google import genai
from openai import OpenAI

from stylo_flora import Snippet
from stylo_flora.benchmarks import benchmark_factory
from stylo_flora.transformer.stylex import StyleX

SRC_PATH = Path('~/playground/research/samples/src.java').expanduser()
CHOICE_DICT_PATH = Path('~/playground/research/samples/choices.json').expanduser()


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
      if isinstance(v, int) and datetime.fromtimestamp(v).year == 2025:
          d[k] = datetime.fromtimestamp(v).strftime('%Y-%m-%d %H:%M:%S')
  return d


def _load_benchmark(args: Namespace) -> list[Snippet]:
  benchmark = benchmark_factory(args.dataset)
  task_to_dataset = {
      'code_translation': 'translation',
      'code_repair': 'repair',
      'code2tag': 'tagging',
      'descode2tag': 'tagging',
      'code_summarization': 'summarization',
      'input_reasoning': 'io_reasoning',
      'output_reasoning': 'io_reasoning',
      'mcq_answering': 'mcq_answering',
      'test_generation': 'test_generation',
  }
  func_name = f'load_for_{task_to_dataset[args.task]}'
  if args.task == 'code_translation':
    return getattr(benchmark, func_name)(args.src_lang, args.dst_lang)
  return getattr(benchmark, func_name)(args.src_lang)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
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
  args = parser.parse_args()

  stylex = StyleX(lang=args.src_lang)
  snippets = _load_benchmark(args)
