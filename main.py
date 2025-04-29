"""
Usage: python3 main.py [OPTIONS]...
"""

from dotenv import load_dotenv
from pprint import pformat

load_dotenv()

from src.evaluate import evaluate
from src.extract import extract_source
from src.transform import transform_source
from src.translate import load_model, translate_with_model
from src.utils import average_codebleu_score, logger, parse_args


def main():
  """
  Assesses the robustness of code translation models by the following steps:

  1. Extracts source code from different datasets into unified data structure.

  2. Applies transformations to the source code to generate a set of transformed code with a code style transformer.

  3. Translates snippets in code set with code translation model.

  4. Evaluates the space spanned by the translated code relative to the original source code.
  """
  args = parse_args()
  for dataset in args.datasets:
    snippets = extract_source(dataset, args.src_lang, args.dst_lang)
    if args.num_snippets >= 0:
      snippets = snippets[:args.num_snippets]
    variants = transform_source(snippets, args.src_lang)
    similarity = average_codebleu_score(snippets, variants, args.src_lang)
    logger.info(f'Average code similarity:\n{pformat(similarity)}')
    translator = load_model(args.model, gpu_id=args.gpu_id)
    translated_snippets = translate_with_model(translator, snippets, args.src_lang, args.dst_lang)
    translated_variants = translate_with_model(translator, variants, args.src_lang, args.dst_lang)
    evaluate(dataset, snippets, variants, translated_snippets, translated_variants, args.src_lang, args.dst_lang)


if __name__ == '__main__':
  main()
