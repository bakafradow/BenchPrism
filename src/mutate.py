import sys
from typing import Sequence

import jpype as jp
import yaml

from . import Snippet

with open('config/settings.yaml', 'r') as f:
  config = yaml.safe_load(f)['mutator']


class Mutator():
  def __init__(self, src_lang, style_file):
    jp.startJVM('-ea', jvmpath=config['jvmpath'],
                classpath=[config['classpath']])
    self.src_lang = src_lang
    self.instance = jp.JClass(config['class']).createMutator(src_lang, style_file)
    if not self.instance:
      raise ValueError(f'Failed to create a mutator with {style_file} in {src_lang}.')

  def __del__(self):
    jp.shutdownJVM()

  def apply(self, snippets: Sequence[Snippet]) -> Sequence[Snippet]:
    mutants = [None] * len(snippets)
    for i, snippet in enumerate(snippets):
      mutant = self.instance.apply(snippet.code)
      if not mutant:
        print(f'Failed to apply mutation to {snippet.id}.', file=sys.stderr)
        mutants[i] = snippet
      else:
        mutants[i] = snippet._replace(code=str(mutant))
    return mutants


def mutate_source(snippets: Sequence[Snippet], src_lang: str) -> Sequence[Snippet]:
  """
  Applies transformations to the source code and generates mutant sequence.
  :param snippets: the snippets to be transformed
  :param src_lang: source language
  :return: the mutant sequence
  """
  print('Applying transformations...')
  style_file = generate_styles(src_lang)
  # TODO: multi-threading optimization
  mutator = Mutator(src_lang, style_file)
  return mutator.apply(snippets)


def generate_styles(src_lang: str) -> str:
  """
  Generates rules for code transformation.
  :param src_lang: source language
  :return: the path to the rules file
  """
  # TODO: generate a style file in XML based on rules.yaml
  return config['style_file']
