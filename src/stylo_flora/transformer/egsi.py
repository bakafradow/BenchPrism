import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor

import jpype as jp
import pandas as pd
import yaml
from tqdm import tqdm

from .. import Snippet, TestBatch
from ..logger import logger
from ..metrics.correctness import calc_correctness
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

    # for debugging on EGSI
    with open('settings.yml') as f:
       self.dump_dir = Path(yaml.safe_load(f)['metrics']['result_dir']) / 'egsi_dump'
    os.makedirs(self.dump_dir, exist_ok=True)

  def __del__(self):
    jp.shutdownJVM()

  def _apply(
      self,
      snippets: Sequence[Snippet],
      lang: str,
      style_file: str,
  ) -> Sequence[Snippet | None]:
    variants = [None] * len(snippets)
    for i, snippet in tqdm(enumerate(snippets), desc='Transforming', total=len(snippets), leave=False):
      try:
        variant = self.cls.apply(lang, snippet.code, style_file)
      except Exception as e:
        logger.error(f'Error occurred for snippet {snippet.id}.\n{e}')
        variant = None
      if not variant:
        logger.warning(f'Failed to transform snippet {i} ({snippet.id}).')
        variants[i] = snippet
      else:
        variants[i] = snippet._replace(code=str(variant))
    return variants

  def _generate_sequences(
      self,
      lang: str,
      seed: int,
  ) -> Sequence[Sequence[int]]:
    equivalent_counts = dict(self.cls.getEquivalentsCounts(lang))
    with NamedTemporaryFile('w', encoding='utf-8', prefix='model', suffix='.txt', delete=False) as f:
      f.write('\n'.join([f'{k}: {",".join(map(str, range(v)))}' for k, v in equivalent_counts.items()]))
      f.flush()
    try:
      returned = subprocess.run([pict_path, f.name, f'/r:{seed}'], check=True, encoding='utf-8', stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
      logger.error(f'Error occurred while running pict.\n{e}')
      raise e
    os.remove(f.name)
    sequences = [[int(num) for num in line.split()] for line in returned.stdout.splitlines()[1:]]
    return sequences

  def _span_until(
      self,
      snippet: Snippet,
      sequence: Sequence[int],
      test_batch: TestBatch,
      lang: str,
      ensure_correct: bool,
      retry: int = -1,
  ) -> str:
    seq_list = jp.java.util.List.of(*[jp.java.lang.Integer(num) for num in sequence])
    attempt = 0
    while retry < 0 or attempt < retry:
      variant = self.cls.span(lang, snippet.code, seq_list)
      if variant:
        if not ensure_correct:
          return variant
        correctness = calc_correctness([snippet._replace(code=str(variant))], [test_batch], lang)
        if math.isclose(correctness, 1.0):
          seq_list = None
          return variant
      attempt += 1
    seq_list = None
    return ''

  def _span(
      self,
      snippets: Sequence[Snippet],
      test_batches: Sequence[TestBatch],
      lang: str,
      seed: int,
      ensure_correct: bool,
  ) -> Sequence[Sequence[Snippet | None]]:
    seqs = self._generate_sequences(lang, seed)
    fallback_rates = pd.Series([0] * len(seqs), name='fallback_rate')

    def worker(snippet_idx, seq_idx, snippet, seq, test_batch):
      try:
        variant = self._span_until(snippet, seq, test_batch, lang, ensure_correct, retry=config['retry'])
        jp.java.lang.System.gc()
      except Exception as e:
        logger.error(f'Error occurred while spanning:\n{e}')
        variant = None
      if not variant:
        logger.warning(f'Failed to transform snippet {snippet_idx} ({snippet.id}) with sequence {seq_idx} ({seq}).')
        with open(self.dump_dir / f'snippet{snippet_idx}_seq{seq_idx}.txt', 'w', encoding='utf-8') as f:
          f.write(f'// Seq={seq}\n\n{snippet.code}')
        fallback_rates[seq_idx] += 1
        return snippet
      logger.debug(f'Successfully transformed snippet {snippet_idx} ({snippet.id}).')
      return snippet._replace(code=str(variant))

    max_workers = max(1, config['max_workers'])
    num_seq = len(seqs)
    corpus = [None] * len(snippets)
    for i, snippet in tqdm(enumerate(snippets), desc='Spanning', total=len(snippets), leave=False):
      with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(worker, [i] * num_seq, range(num_seq),
                                         [snippet] * num_seq, seqs,
                                         [test_batches[i] for _ in range(num_seq)]),
                            desc=f'Spanning snippet {i}', total=num_seq, leave=False))
      corpus[i] = results
    logger.info(f'Spanned {len(corpus)} variant benchmarks.')

    series = pd.Series(fallback_rates / len(snippets), name='fallback_rate')
    series.to_csv(self.dump_dir / 'fallback_rates.csv', index=False)

    return corpus

  def transform(
      self,
      snippets: Sequence[Snippet],
      test_batches: Sequence[TestBatch],
      lang: str,
      *,
      seed: int = 42,
      ensure_correct: bool = True,
  ) -> Sequence[Sequence[Snippet | None]]:
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
      return self._span(snippets, test_batches, lang, seed, ensure_correct)
    logger.info(f'Applying styles from {style_file}...')
    return [self._apply(snippets, lang, style_file)]
