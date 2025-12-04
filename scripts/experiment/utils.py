from argparse import Namespace

from stylo_flora import Snippet
from stylo_flora.benchmarks import benchmark_factory


def load_snippets(args: Namespace) -> list[Snippet]:
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
