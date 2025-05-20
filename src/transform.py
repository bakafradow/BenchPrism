import os
import shutil
import subprocess
from typing import Sequence
from tempfile import NamedTemporaryFile

import jpype as jp
import yaml
from tqdm import tqdm

from . import Snippet
from .utils import logger

with open('settings.yml', 'r') as f:
  config = yaml.safe_load(f)['mutator']

pict_path = os.getenv('PICT_PATH')
if not pict_path or not shutil.which(pict_path):
  raise ValueError(f'PICT_PATH is not set or the pict executable is not found at {pict_path}.')


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
    equivalent_counts = dict(self.cls.getEquivalentsCounts(self.lang))
    with NamedTemporaryFile('w', encoding='utf-8', prefix='model', suffix='.txt', delete=False) as f:
      f.write('\n'.join([f'{k}: {",".join(map(str, range(v)))}' for k, v in equivalent_counts.items()]))
      f.flush()
    try:
      returned = subprocess.run([pict_path, f.name], check=True, encoding='utf-8', stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
      logger.error(f'Error occurred while running pict:\n{e}')
      raise e
    os.remove(f.name)
    sequences = [[int(num) for num in line.split()] for line in returned.stdout.splitlines()[1:]]
    corpus = [None] * len(sequences)
    for i, sequence in tqdm(enumerate(sequences), desc='Spanning', total=len(sequences), leave=False):
      variants = [None] * len(snippets)
      for j, snippet in tqdm(enumerate(snippets), desc='Transforming', total=len(snippets), leave=False):
        try:
          sequence_list = jp.java.util.List.of(*[jp.java.lang.Integer(num) for num in sequence])
          variant = self.cls.span(self.lang, snippet.code, sequence_list)
        except Exception as e:
          logger.error(f'Error occurred for snippet {snippet.id}:\n{e}')
          variant = None
        if not variant:
          logger.warning(f'Failed to transform {snippet.id}.')
          variants[j] = snippet
        else:
          variants[j] = snippet._replace(code=str(variant))
      corpus[i] = variants
    return corpus


def transform_source(snippets: Sequence[Snippet], src_lang: str) -> Sequence[Sequence[Snippet]]:
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
  return [mutator.apply(src_lang, style_file)]
