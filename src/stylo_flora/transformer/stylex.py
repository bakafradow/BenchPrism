import atexit
import inspect
import math
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping
from collections.abc import MutableSequence as MSeq
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, ParamSpec, TypeVar

import jpype as jp
import jpype.imports
import yaml
from tqdm import tqdm


def shutdown():
  if jp.isJVMStarted():
    jp.shutdownJVM()
    logger.info('JVM shutdown successfully.')


jp.startJVM('-ea', '--enable-native-access=ALL-UNNAMED')
atexit.register(shutdown)

from org.example import Configuration
from org.example.controller import (Applicator, Extractor, StylerContainer,
                                    TokenAugmentor)
from org.example.myException import ApplyException, ExtractException
from org.example.parser.common import MyParseTreeWalker
from org.example.parser.common.factory import MyParserFactory
from org.example.parser.java import SpotDetectorListener
from org.example.style import ProgramStyle, StyleFileIO
from org.example.styler import Stage
GlobalInfo = jp.JClass('org.example.global.GlobalInfo')  # cannot import directly due to package name

from .. import Snippet
from ..logger import logger
from ..metrics.correctness import calc_correctness
from .base import BaseTransformer
from .stylex_builders import build_styler

with open('configs/settings.yaml', 'r') as f:
  config = yaml.safe_load(f)['transformer']
with open('configs/stylex_options.yaml', 'r') as f:
  option_config = yaml.safe_load(f)

if not shutil.which('pict'):
  raise ValueError('PICT executable not found.')

P = ParamSpec('P')
T = TypeVar('T')


def set_global_info(func: Callable[P, T]) -> Callable[P, T]:
  """
  A decorator to set global configuration and language before executing the function, which is a boilerplate for StyleX.
  """
  def wrapper(*args, **kwargs):
    try:
      sig = inspect.signature(func)
      bound_args = sig.bind(*args, **kwargs)
      bound_args.apply_defaults()
      lang = bound_args.arguments['lang']
    except TypeError as e:
      raise TypeError(f'Failed to bind arguments for {func.__name__}.\n{e}') from e
    except KeyError as e:
      raise TypeError(f'Decorator @set_global_info requires {func.__name__} to have a \'lang\' argument.') from e
    GlobalInfo.setConf(Configuration())
    GlobalInfo.setLanguage(lang)
    return func(*args, **kwargs)
  return wrapper


class StyleX(BaseTransformer):
  lock = Lock()

  @set_global_info
  def transform(
      self,
      snippets: Seq[Snippet],
      lang: str,
      **kwargs,
  ) -> list[list[Snippet | None]]:
    """
    Applies transformations to the source code and generates variant sequence.
    :param snippets: the snippets to be transformed
    :param lang: the language of the snippets
    :param seed: the random seed for reproducibility
    :param ensure_correct: whether to reject incorrectly transformed snippets
    :return: a series of transformed snippets, each of which is corresponding to a variant sequence
    """
    seed = kwargs.get('seed', 42)
    ensure_correct = kwargs.get('ensure_correct', True)

    def worker(
        snippet_idx: int,
        snippet: Snippet,
        seq_idx: int,
        styler_container: jp.JObject,
    ) -> Snippet | None:
      if seq_idx in snippet.args.get('transformed_seqs', set()):
        return None
      try:
        with self.lock:
          variant_code = self._apply_styles(lang, snippet.code, styler_container)
        if variant_code and ensure_correct:
          correctness = calc_correctness([variant_code], [snippet.args], lang)
          if not math.isclose(correctness, 1.0):
            logger.warning(f'Correctness check failed: {correctness}')
            variant_code = None
      except jp.JVMNotRunning:  # in case of keyboard interrupt
        return None
      except Exception as e:
        logger.error(f'{e.__class__.__name__} occurred while spanning:\n{e}')
        variant_code = None
      if not variant_code:
        logger.warning(f'Failed to transform snippet {snippet_idx} ({snippet.id}) with sequence {seq_idx} ({seqs[seq_idx]}).')
        return None
      logger.debug(f'Successfully transformed snippet {snippet_idx} ({snippet.id}).')
      return snippet.replace(code=str(variant_code))

    option_counts = self._count_options()
    seqs = self._generate_seqs(seed, option_counts)
    num_seq = len(seqs)

    styler_containers = []
    for seq in seqs:
      choice_dict = self._create_choice_dict(seq)
      styler_container = self._build_styler_container(lang, choice_dict)
      styler_containers.append(styler_container)

    corpus = []
    for i, snippet in tqdm(enumerate(snippets), desc='Spanning', total=len(snippets), leave=False):
      with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        results = list(tqdm(executor.map(track_time(worker), [i] * num_seq, [snippet] * num_seq,
                                         range(num_seq), styler_containers),
                            desc=f'Spanning snippet {i}', total=num_seq, leave=False))
      corpus.append(results)
    logger.info(f'Spanned {len(corpus)} variant benchmarks.')
    return corpus

  @set_global_info
  def count_spots(
      self,
      snippets: Seq[Snippet],
      lang: str,
  ) -> list[int]:
    counts = []
    for i, snippet in enumerate(snippets):
      spots = jp.java.util.HashMap()
      parser = MyParserFactory.createParser(lang)
      listener = SpotDetectorListener(spots, parser)
      tree = parser.parseFromString(snippet.code)
      walker = MyParseTreeWalker()
      walker.walk(listener, tree)
      counts.append(spots.size())
    return counts

  def _extract_from_file(
      self,
      lang: str,
      style_file: str,
  ) -> jp.JObject:
    parser = MyParserFactory.createParser(lang)
    program_style = StyleFileIO.read(style_file, parser)
    container = StylerContainer()
    for styler in container.getStylers():
      style = program_style.getStyle(styler.getStyle().getStyleName())
      if style:
        styler.setStyle(style)
    return container

  def _extract_from_code(
      self,
      lang: str,
      code: str,
  ) -> jp.JObject:
    parser = MyParserFactory.createParser(lang)
    if not parser.parseFromString(code):
      logger.warning('Compilation error.')
      return None
    container = StylerContainer()
    token_augmentor = TokenAugmentor()
    try:
      Extractor.extractRules(parser, container, token_augmentor)
    except ExtractException as e:
      logger.warning(f'Failed to extract rules.\n{e}')
      return None
    program_style = ProgramStyle()
    for styler in container.getStylers():
      if styler.isEnable(Stage.EXTRACT):
        styler.extractFinalize()
      program_style.add(styler.getStyle())
    return program_style

  def _apply_styles(
      self,
      lang: str,
      code: str,
      styler_container: jp.JObject,
  ) -> str | None:
    try:
      parser = MyParserFactory.createParser(lang)
      if not parser.parseFromString(code):
        logger.warning('Compilation error.')
        return None
      token_augmentor = TokenAugmentor()
      tokens = Applicator.applyRules(parser, styler_container, token_augmentor)
      if tokens[-1].getType() == parser.getEOF():
        tokens.remove(tokens.size() - 1)  # remove EOF token
      token_augmentor.restoreState(tokens, parser)
      return ''.join([str(token.getText()) for token in tokens])
    except ApplyException as e:
      logger.warning(f'Failed to apply rules.\n{e}')
      return None

  def _count_options(
      self,
  ) -> list[int]:
    option_counts = []
    for style in chain(*[list(item.values())[0] for item in option_config]):
      option = list(style.values())[0]
      match option.get('option_type'):
        case 'bool':
          count = 2
        case 'number':
          count = option.get('length', 0)
        case 'enum':
          count = len(option.get('options', []))
        case _:
          raise ValueError(f'Unknown option type: {option.get("option_type")}')
      option_counts.append(count)
    return option_counts

  def _generate_seqs(
      self,
      seed: int,
      option_counts: Seq[int],
  ) -> list[list[int]]:
    with NamedTemporaryFile('w', encoding='utf-8', prefix='model', suffix='.txt', delete=False) as f:
      f.write('\n'.join([f'{i}: {",".join(map(str, range(count)))}' for i, count in enumerate(option_counts)]))
      f.flush()
    try:
      args = ['pict', f.name, f'/r:{seed}']
      completed = subprocess.run(args, check=True, encoding='utf-8', stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
      logger.error(f'{e.__class__.__name__} occurred while running pict.\n{e}')
      raise e
    os.remove(f.name)
    seqs = [[int(num) for num in line.split()] for line in completed.stdout.splitlines()[1:]]
    return seqs

  def _create_choice_dict(
      self,
      seq: Seq[int],
  ) -> Mapping[str, Mapping[str, Any]]:
    choice_dict: dict[str, dict[str, Any]] = defaultdict(dict)
    idx = 0
    for item in option_config:
      styler_name = list(item.keys())[0]
      styles = item[styler_name]
      for i in range(len(styles)):
        style_name = list(styles[i].keys())[0]
        option_item = styles[i][style_name]
        match option_item.get('option_type'):
          case 'bool':
            choice = seq[idx] == 1
          case 'number':
            choice = seq[idx]  # type: ignore[assignment]
          case 'enum':
            choice = option_item.get('options')[seq[idx]]
          case _:
            raise ValueError(f'Unknown option type: {option_item.get("option_type")}')
        choice_dict[styler_name][style_name] = choice
        idx += 1
    return choice_dict

  def _build_styler_container(
      self,
      lang: str,
      choice_dict: Mapping[str, Mapping[str, Any]],
  ) -> jp.JObject:
    styler_container = StylerContainer()
    for styler in styler_container.getStylers():
      if not styler.isEnable(Stage.APPLY):
        continue
      styler_name = styler.getClass().getSimpleName()
      try:
        choices = choice_dict[styler_name]
        build_styler(styler, lang, choices)
        styler.getStyle().fillStyle()
      except KeyError:
        logger.warning(f'No choices found for styler {styler_name}. Skipping.')
      except Exception as e:
        logger.warning(f'{e.__class__.__name__} occurred while building styler {styler_name}:\n{e}')
    return styler_container
