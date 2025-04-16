import argparse
import re
from itertools import groupby
from operator import itemgetter
from pprint import pformat
from random import sample
from typing import NamedTuple, Sequence

import jsonlines

from src import Snippet
from src.evaluate import evaluate
from src.translate import load_model, translate_with_model
from src.utils import average_codebleu_score, logger, parse_args


class Arguments(NamedTuple):
  model: str
  src_lang: str
  dst_lang: str
  generator: str
  transformer: str
  gpu_id: int = -1
  num_snippets: int = -1


def parse_args() -> Arguments:
  default_src_lang = 'java'
  default_dst_lang = 'cpp'
  generators = ['claude35sonnet', 'deepseekcoder', 'gpt4o', 'wizardcoder', 'human']
  transformers = ['claude35sonnet', 'deepseekcoder', 'gpt4o', 'wizardcoder', 'egsi', 'codebuff']
  parser = argparse.ArgumentParser(description='Code translation evaluation tool.'
                                               'All the datasets are evaluated by default.')
  parser.add_argument('-m', '--model', type=str,
                      required=True,
                      help='Specify the model to use.')
  parser.add_argument('--src-lang', default=default_src_lang, type=str,
                      choices=['java', 'cpp'],
                      help=f'Specify the source language, {default_src_lang} by default.')
  parser.add_argument('--dst-lang', default=default_dst_lang,type=str,
                      choices=['c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust'],
                      help=f'Specify the destination language, {default_dst_lang} by default.')
  parser.add_argument('-g', '--generator', type=str,
                      required=True, choices=generators,
                      help='Specify the code generator.')
  parser.add_argument('-t', '--transformer', type=str,
                      required=True, choices=transformers,
                      help='Specify the code transformer.')
  parser.add_argument('-i', '--gpu-id', type=int, default=-1,
                      help='Specify the GPU to use.')
  parser.add_argument('-n', '--num-snippets', type=int, default=-1,
                      help='Limit the number of snippets to test.')
  args = parser.parse_args()
  return Arguments(
    model=args.model,
    src_lang=args.src_lang,
    dst_lang=args.dst_lang,
    generator=args.generator,
    transformer=args.transformer,
    gpu_id=args.gpu_id,
    num_snippets=args.num_snippets,
  )


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
  dataset = 'CodeNet'
  logger.info(f'Evaluating {dataset} from {args.generator} to {args.transformer}...')
  snippets = _load_snippets(args.generator, args.transformer)
  variants = _load_variants(args.generator, args.transformer, snippets)
  snippets = [snippet for snippet, variant in zip(snippets, variants) if variant is not None]
  variants = [variant for variant in variants if variant is not None]
  if args.num_snippets >= 0:
    snippets = snippets[:args.num_snippets]
    variants = variants[:args.num_snippets]
  similarity = average_codebleu_score(snippets, variants, args.src_lang)
  logger.info(f'Average code similarity:\n{pformat(similarity)}')
  translator = load_model(args.model, gpu_id=args.gpu_id)
  translated_snippets = translate_with_model(translator, snippets, args.src_lang, args.dst_lang)
  translated_variants = translate_with_model(translator, variants, args.src_lang, args.dst_lang)
  evaluate(dataset, snippets, variants, translated_snippets, translated_variants, args.src_lang, args.dst_lang)


if __name__ == '__main__':
  main()
