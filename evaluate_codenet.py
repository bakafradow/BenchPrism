from pprint import pformat
from typing import Sequence

import jsonlines

from src import Snippet
from src.evaluate import evaluate
from src.translate import load_model, translate_with_model
from src.utils import average_codebleu_score, logger, parse_args


def _load_snippets(path: str) -> Sequence[Snippet]:
  with jsonlines.open(path) as reader:
    return [Snippet(snippet['id'], snippet['code']) for snippet in reader]


def _load_variants(path: str, snippets: Sequence[Snippet], generator: str) -> Sequence[Snippet]:
  with jsonlines.open(path) as reader:
    objects = [obj for obj in reader if obj['src']['author_name'] == generator]
  variants = [None] * len(snippets)
  for i, snippet in enumerate(snippets):
    target = next((obj for obj in objects if obj['src']['problem_id'] == snippet.id), None)
    if target:
      variants[i] = Snippet(snippet.id, target['result']['file_name'])
  return variants


def main():
  args = parse_args()
  dataset = 'CodeNet'
  generator = 'claude35sonnet'
  transformer = 'deepseekcoder'
  logger.info(f'Evaluating {dataset} from {generator} to {transformer}...')
  snippets = _load_snippets(f'data/CodeNet/dataset/codenet/{generator}_codenet_in_out.jsonl')
  variants = _load_variants(f'data/CodeNet/result/codenet_{transformer}.jsonl', snippets, generator)
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
