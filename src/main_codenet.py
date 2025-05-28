import argparse
from collections.abc import Sequence
from itertools import groupby
from operator import itemgetter
from pprint import pformat
from random import sample

import jsonlines
from dotenv import load_dotenv

load_dotenv()

from . import Snippet
from .agent.base import agent_factory
from .benchmarks import benchmark_factory
from .logger import init_logger, logger
from .main import evaluate_translation
from .metrics.similarity import calculate_codebleu


def parse_args() -> argparse.Namespace:
  generators = ['claude35sonnet', 'deepseekcoder', 'gpt4o', 'wizardcoder', 'human']
  transformers = ['claude35sonnet', 'deepseekcoder', 'gpt4o', 'wizardcoder', 'egsi', 'codebuff']
  parser = argparse.ArgumentParser(description='Code translation evaluation tool.'
                                               'All the datasets are evaluated by default.')
  parser.add_argument('-m', '--model', type=str,
                      required=True,
                      help='Specify the model to use.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('--dst-lang', type=str, required=True,
                      help='Specify the destination language.')
  parser.add_argument('-g', '--generator', type=str,
                      required=True, choices=generators,
                      help='Specify the code generator.')
  parser.add_argument('-t', '--transformer', type=str,
                      required=True, choices=transformers,
                      help='Specify the code transformer.')
  parser.add_argument('--device', type=str, default='auto',
                      help='Specify the device to use for LLM inference. If set to "auto", it will use the largest available GPU, or CPU if no GPU is available.')
  parser.add_argument('-n', '--num-snippets', type=int, default=-1,
                      help='Limit the number of snippets to test. -1 for all.')
  parser.add_argument('-v', '--verbose', action='store_true', default=False,
                      help='If set, enables verbose level logging.')
  parser.add_argument('--debug', action='store_true', default=False,
                      help='If set, enables debugging level logging.')
  args = parser.parse_args()
  return args


def _load_snippets(generator: str, transformer: str) -> Sequence[Snippet]:
  if generator == 'human':
    with jsonlines.open(f'data/CodeNet/result/codenet_{transformer}.jsonl') as reader:
      authors = sorted((obj['src']['problem_id'], obj['src']['author_name']) for obj in reader)
    id_to_authors = {id_: set(map(itemgetter(1), pairs)) for id_, pairs in groupby(authors, key=itemgetter(0))}
    with jsonlines.open('data/CodeNet/dataset/codenet/human_codenet_solution.jsonl') as reader:
      solutions = sorted((obj['id'], obj['code'], obj['author_name']) for obj in reader
                         if obj['id'] in id_to_authors and obj['author_name'] in id_to_authors[obj['id']])
    samples = [sample(list(triplets), 1)[0] for _, triplets in groupby(solutions, key=itemgetter(0))]
    return [Snippet(*triplet) for triplet in samples]
  with jsonlines.open(f'data/CodeNet/dataset/codenet/{generator}_codenet_in_out.jsonl') as reader:
    return [Snippet(snippet['id'], snippet['code']) for snippet in reader]


def _load_variants(generator: str, transformer: str, snippets: Sequence[Snippet]) -> Sequence[Snippet]:
  with jsonlines.open(f'data/CodeNet/result/codenet_{transformer}.jsonl') as reader:
    if generator == 'human':
      objects = list(reader)
    else:
      objects = [obj for obj in reader if obj['src']['author_name'] == generator]
  variants = [None] * len(snippets)
  for i, snippet in enumerate(snippets):
    target = next((obj for obj in objects if obj['src']['problem_id'] == snippet.id and (generator != 'human' or obj['src']['author_name'] == snippet.ref)), None)
    if target:
      variants[i] = Snippet(snippet.id, target['result']['file_name'])
  return variants


def main():
  args = parse_args()
  init_logger(verbose=args.verbose, debug=args.debug)
  dataset = 'CodeNet'
  benchmark = benchmark_factory(dataset)
  translator = agent_factory(args.model, device=args.device)
  logger.info(f'Evaluating {args.model} on {args.dataset} with generator {args.generator} and transformer {args.transformer}...')

  snippets = _load_snippets(args.generator, args.transformer)
  variants = _load_variants(args.generator, args.transformer, snippets)
  snippets = [snippet for snippet, variant in zip(snippets, variants) if variant is not None]
  variants = [variant for variant in variants if variant is not None]
  if args.num_snippets >= 0:
    snippets = snippets[:args.num_snippets]
    variants = variants[:args.num_snippets]

  similarity = calculate_codebleu(snippets, variants, args.src_lang)
  logger.info(f'Average code similarity:\n{pformat(similarity)}')

  corpus = (variants,)
  evaluate_translation(benchmark, translator, snippets, corpus, args)


if __name__ == '__main__':
  main()
