import re
from argparse import Namespace

from benchprism import Snippet
from benchprism.benchmarks import benchmark_factory


def get_data_id(args: Namespace) -> str:
  """
  Constructs a unique identifier for specific snippet series in a benchmark.
  """
  return f'{args.dataset.lower()}_{args.task}_{args.src_lang}_seed{args.seed}'


def get_eval_id(args: Namespace) -> str:
  """
  Constructs a unique identifier for an experiment set.
  """
  eval_id = f'{args.dataset.lower()}_{args.task}_{args.src_lang}'
  if args.task == 'code_translation':
    eval_id += f'{"_to_" + args.dst_lang}'
  eval_id += f'_with_{re.split(r"[:/]", args.model)[-1]}_seed{args.seed}'
  return eval_id


def load_snippets(args: Namespace) -> list[Snippet]:
  benchmark = benchmark_factory(args.dataset)
  task_to_dataset = {
      'code_translation': 'translation',
      'code_repair': 'repair',
      'code2tag': 'tagging',
      'descode2tag': 'tagging',
      'test_generation': 'test_generation',
      'code_summarization': 'summarization',
      'mcq_answering': 'mcq_answering',
      'input_reasoning': 'io_reasoning',
      'output_reasoning': 'io_reasoning',
      'defect_detection': 'defect_detection',
  }
  func_name = f'load_for_{task_to_dataset[args.task]}'
  if args.task == 'code_translation':
    return getattr(benchmark, func_name)(args.src_lang, args.dst_lang)
  return getattr(benchmark, func_name)(args.src_lang)
