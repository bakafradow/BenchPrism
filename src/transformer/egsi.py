import os
import shutil
import subprocess
from collections.abc import Sequence
from tempfile import NamedTemporaryFile

import jpype as jp
import yaml
from tqdm import tqdm

from .. import Snippet
from ..logger import logger
from .base import BaseTransformer

with open('settings.yml', 'r') as f:
  config = yaml.safe_load(f)['transformer']

pict_path = os.getenv('PICT_PATH', 'pict')
if not pict_path or not shutil.which(pict_path):
  raise ValueError(f'PICT_PATH is not set or the pict executable is not found at {pict_path}.')


class EGSI(BaseTransformer):
  def __init__(self):
    jp.startJVM('-ea', jvmpath=os.getenv('JVM_PATH'),
                classpath=[os.getenv('TSFM_CLASSPATH')])
    self.cls = jp.JClass(config['class'])
    if not self.cls:
      raise ValueError('Failed to load the transformer class.')

  def __del__(self):
    jp.shutdownJVM()

  def _apply(self, snippets: Sequence[Snippet], lang: str, style_file: str) -> Sequence[Snippet | None]:
    variants = [None] * len(snippets)
    for i, snippet in tqdm(enumerate(snippets), desc='Transforming', total=len(snippets), leave=False):
      try:
        variant = self.cls.apply(lang, snippet.code, style_file)
      except Exception as e:
        logger.error(f'Error occurred for snippet {snippet.id}:\n{e}')
        variant = None
      if not variant:
        logger.warning(f'Failed to transform to {snippet.id}.')
        variants[i] = snippet
      else:
        variants[i] = snippet._replace(code=str(variant))
    return variants

  def _span(self, snippets: Sequence[Snippet], lang: str, seed: int) -> Sequence[Sequence[Snippet | None]]:
    equivalent_counts = dict(self.cls.getEquivalentsCounts(lang))
    with NamedTemporaryFile('w', encoding='utf-8', prefix='model', suffix='.txt', delete=False) as f:
      f.write('\n'.join([f'{k}: {",".join(map(str, range(v)))}' for k, v in equivalent_counts.items()]))
      f.flush()
    try:
      returned = subprocess.run([pict_path, f.name, f'/r:{seed}'], check=True, encoding='utf-8', stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
      logger.error(f'Error occurred while running pict:\n{e}')
      raise e
    os.remove(f.name)
    sequences = [[int(num) for num in line.split()] for line in returned.stdout.splitlines()[1:]]
    corpus = [[None] * len(snippets) for _ in range(len(sequences))]
    for i, sequence in tqdm(enumerate(sequences), desc='Spanning', total=len(sequences), leave=False):
      for j, snippet in tqdm(enumerate(snippets), desc='Transforming', total=len(snippets), leave=False):
        try:
          sequence_list = jp.java.util.List.of(*[jp.java.lang.Integer(num) for num in sequence])
          variant = self.cls.span(lang, snippet.code, sequence_list)
        except Exception as e:
          logger.error(f'Error occurred for snippet {snippet.id}:\n{e}')
          variant = None
        if not variant:
          logger.warning(f'Failed to transform {snippet.id}.')
          corpus[i][j] = snippet
        else:
          corpus[i][j] = snippet._replace(code=str(variant))
    logger.info(f'Spanned {len(corpus)} variant benchmarks.')
    return corpus

  def transform(self, snippets: Sequence[Snippet], lang: str, *, seed: int = 42) -> Sequence[Sequence[Snippet | None]]:
    """
    Applies transformations to the source code and generates variant sequence.
    :param snippets: the snippets to be transformed
    :param lang: the language of the snippets
    :param seed: the random seed for reproducibility
    :return: the variant sequence
    """
    style_file = os.getenv('STYLE_FILE')
    if not style_file:
      logger.info('Spanning styles...')
      return self._span(snippets, lang, seed)
    logger.info(f'Applying styles from {style_file}...')
    return [self._apply(snippets, lang, style_file)]
