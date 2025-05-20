import os
from itertools import chain
from typing import Sequence

import jpype as jp
import yaml
from tqdm import tqdm

from . import Snippet
from .utils import logger

with open('settings.yml', 'r') as f:
  config = yaml.safe_load(f)['mutator']


class Mutator():
  def __init__(self, lang):
    jp.startJVM('-ea', jvmpath=os.getenv('JVM_PATH'),
                classpath=[os.getenv('TSFM_CLASSPATH')])
    self.lang = lang
    self.cls = jp.JClass(config['class'])
    if not self.cls:
      raise ValueError('Failed to load the mutator class.')

  def __del__(self):
    jp.shutdownJVM()

  def apply(self, snippets: Sequence[Snippet], style_file: str) -> Sequence[Snippet]:
    variants = [None] * len(snippets)
    for i, snippet in tqdm(enumerate(snippets), desc='Transforming', total=len(snippets), leave=False):
      try:
        variant = self.cls.apply(self.lang, snippet.code, style_file)
      except Exception as e:
        logger.error(f'Error occurred for snippet {snippet.id}:\n{e}')
        variant = None
      if not variant:
        logger.warning(f'Failed to transform to {snippet.id}.')
        variants[i] = snippet
      else:
        variants[i] = snippet._replace(code=str(variant))
    return variants

  def span(self, snippets: Sequence[Snippet]) -> Sequence[Sequence[Snippet]]:
    variants_lists = [None] * len(snippets)
    for i, snippet in tqdm(enumerate(snippets), desc='Spanning', total=len(snippets), leave=False):
      try:
        variants = self.cls.span(self.lang, snippet.code)
      except Exception as e:
        logger.error(f'Error occurred for snippet {snippet.id}:\n{e}')
        variants = []
      if not variants:
        logger.warning(f'Failed to span {snippet.id}.')
        variants_lists[i] = [snippet]
      else:
        variants_lists[i] = [snippet._replace(code=str(variant)) for variant in variants]
    return list(chain.from_iterable(variants_lists))


def transform_source(snippets: Sequence[Snippet], src_lang: str) -> Sequence[Snippet]:
  """
  Applies transformations to the source code and generates variant sequence.
  :param snippets: the snippets to be transformed
  :param src_lang: source language
  :return: the variant sequence
  """
  # TODO: load existing variants if available
  # TODO: multi-threading optimization
  mutator = Mutator(src_lang)
  style_file = os.getenv('STYLE_FILE')
  if not style_file:
    logger.info('Spanning styles...')
    return mutator.span(snippets)
  logger.info(f'Applying styles from {style_file}...')
  return mutator.apply(src_lang, style_file)
