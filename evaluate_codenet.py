from typing import Sequence

import jsonlines

from src import Snippet
from src.evaluate import evaluate
from src.translate import load_model, translate_with_model
from src.utils import parse_args


def _load_snippets(path: str) -> Sequence[Snippet]:
  with jsonlines.open(path) as reader:
    return [Snippet(snippet['id'], snippet['code']) for snippet in reader]


def _load_mutants(path: str, snippets: Sequence[Snippet]) -> Sequence[Snippet]:
  with jsonlines.open(path) as reader:
    objects = [obj for obj in reader]
  mutants = [None] * len(snippets)
  for i, snippet in enumerate(snippets):
    target = next((obj for obj in objects if obj['src']['problem_id'] == snippet.id), None)
    if target:
      mutants[i] = Snippet(snippet.id, target['result']['file_name'])
    else:
      mutants[i] = Snippet(snippet.id, snippet.code)
  return mutants


def main():
  args = parse_args()
  for dataset in args.datasets:
    snippets = _load_snippets('data/CodeNet/dataset/codenet/gpt4o_codenet_in_out.jsonl')
    mutants = _load_mutants('data/CodeNet/result/codenet_claude35sonnet.jsonl', snippets)
    if args.num_snippets >= 0:
      snippets = snippets[:args.num_snippets]
      mutants = mutants[:args.num_snippets]
    translator = load_model(args.model, gpu_id=args.gpu_id)
    translated_snippets = translate_with_model(translator, snippets, args.src_lang, args.dst_lang)
    translated_mutants = translate_with_model(translator, mutants, args.src_lang, args.dst_lang)
    evaluate(dataset, snippets, mutants, translated_snippets, translated_mutants, args.src_lang, args.dst_lang)


if __name__ == '__main__':
  main()
