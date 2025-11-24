import atexit
import math
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence as Seq
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import jpype as jp
import jpype.imports
import yaml
from tqdm import tqdm

jp.startJVM('-ea', '--enable-native-access=ALL-UNNAMED')

from .stylex_builders import build_styler
from .base import BaseTransformer
from ..metrics.correctness import pass_at_1
from ..logger import logger
from .. import Snippet, setting_dict

from org.example.styler import Stage
from org.example.style import ProgramStyle, StyleFileIO
from org.example.parser.java import SpotDetectorListener
from org.example.parser.common.factory import MyParserFactory
from org.example.parser.common import MyParseTreeWalker
from org.example.myException import ApplyException, ExtractException
from org.example.controller import (Applicator, Extractor, StylerContainer,
                                    TokenAugmentor)
from org.example import Configuration
GlobalInfo = jp.JClass('org.example.global.GlobalInfo')  # cannot import directly due to package name

def shutdown():
  if jp.isJVMStarted():
    jp.shutdownJVM()
    logger.info('JVM shutdown successfully.')
atexit.register(shutdown)

with open('configs/stylex_options.yaml', 'r') as f:
  option_dict = yaml.safe_load(f)

if not shutil.which('pict'):
  raise ValueError('PICT executable not found.')


class StyleX(BaseTransformer):
  def __init__(self, lang: str, seed: int = 42):
    super().__init__()
    self.lang = lang
    GlobalInfo.setConf(Configuration())
    GlobalInfo.setLanguage(lang)

    option_counts = self._count_options()
    seqs = self._generate_seqs(seed, option_counts)
    self.seqs = seqs
    self.styler_containers = self._seq_to_styler_containers(lang, seqs)

    self.lock = Lock()
    self.executor = ThreadPoolExecutor(max_workers=setting_dict['transformer']['max_workers'])

  def transform(
      self,
      snippet: Snippet,
      *,
      seqs_to_skip: set[int] = set(),
  ) -> list[str | None]:
    def worker(seq_idx: int, styler_container: jp.JObject) -> str | None:
      if seq_idx in seqs_to_skip:
        return None
      try:
        with self.lock:
          variant = self._apply_styles(self.lang, snippet.data['code'], styler_container)
        if variant:
          checker = snippet.data.get('checker')
          if checker and not checker(snippet.replace(code=variant), self.lang):
            variant = None
      except jp.JVMNotRunning:  # in case of keyboard interrupt
        return None
      except Exception as e:
        logger.error(f'{e.__class__.__name__} occurred while spanning:\n{e}')
        variant = None
      if not variant:
        logger.warning(f'Failed to transform snippet {snippet.id} with sequence {seq_idx} ({self.seqs[seq_idx]}).')
        return None
      logger.debug(f'Successfully transformed snippet {snippet.id}.')
      return variant

    variants = list(tqdm(self.executor.map(worker, range(len(self.seqs)), self.styler_containers),
                        desc=f'Spanning {snippet.id}', total=len(self.seqs), leave=False))
    return variants

  def is_processable(self, snippet: Snippet) -> bool:
    parser = MyParserFactory.createParser(self.lang)
    return parser.parseFromString(snippet.data['code']) is not None

  def count_spots(
      self,
      snippet: Snippet,
  ) -> int:
    """
    Counts spots applicable for style transformation in the given code snippet.

    :param snippet: the code snippet to analyze
    :return: the number of spots
    """
    spots = jp.java.util.HashMap()
    parser = MyParserFactory.createParser(self.lang)
    listener = SpotDetectorListener(spots, parser)
    tree = parser.parseFromString(snippet.data['code'])
    walker = MyParseTreeWalker()
    walker.walk(listener, tree)
    return spots.size()

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
    parser = MyParserFactory.createParser(lang)
    if not parser.parseFromString(code):
      logger.warning('Compilation error.')
      parser = None
      del parser
      return None
    token_augmentor = TokenAugmentor()
    try:
      tokens = Applicator.applyRules(parser, styler_container, token_augmentor)
    except ApplyException as e:
      logger.warning(f'Failed to apply rules:\n{e}')
      parser = None
      token_augmentor = None
      del parser, token_augmentor
      return None
    if tokens[-1].getType() == parser.getEOF():
      tokens.remove(tokens.size() - 1)  # remove EOF token
    token_augmentor.restoreState(tokens, parser)
    variant = ''.join([str(token.getText()) for token in tokens])
    parser = None
    token_augmentor = None
    tokens = None
    del parser, token_augmentor, tokens
    return variant

  def _count_options(
      self,
  ) -> list[int]:
    option_counts = []
    for style in chain(*[list(item.values())[0] for item in option_dict]):
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

  def _seq_to_styler_containers(
      self,
      lang: str,
      seqs: Seq[Seq[int]],
  ) -> list[jp.JObject]:
    styler_containers = []
    for seq in seqs:
      choice_dict = self._create_choice_dict(seq)
      styler_container = self._build_styler_container(lang, choice_dict)
      styler_containers.append(styler_container)
    return styler_containers

  def _create_choice_dict(
      self,
      seq: Seq[int],
  ) -> Mapping[str, Mapping[str, Any]]:
    choice_dict: dict[str, dict[str, Any]] = defaultdict(dict)
    idx = 0
    for item in option_dict:
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
