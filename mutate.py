import yaml

from mutators.tree_transformer import TreeTransformer
from snippet import SnippetSequence
from mutators.style_transformer import CodeStyleTransformer


def mutate_source(snippets: SnippetSequence, src_lang: str) -> SnippetSequence:
  """
  Applies transformations to the source code to generate a set of mutated code with a code style transformer.
  :param snippets: the snippets to be transformed
  :param src_lang: source language
  :return: a set of code which indicates different combinations of mutations
  """
  print('Applying transformations...')
  with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)['mutator']
  if config['legacy']:
    mutator = CodeStyleTransformer(src_lang=src_lang)
  else:
    mutator = TreeTransformer(src_lang=src_lang)
  return mutator.apply_one(snippets)
