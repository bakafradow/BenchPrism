import atexit
import math
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from tempfile import NamedTemporaryFile

import jpype as jp
import yaml
from tqdm import tqdm

def shutdown():
  if jp.isJVMStarted():
    jp.shutdownJVM()
    logger.info('JVM shutdown successfully.')


jp.startJVM('-ea', jvmpath=os.getenv('JVM_PATH'),
            classpath=[os.getenv('STYLEX_CLASSPATH')])
atexit.register(shutdown)

from .. import Snippet
from ..logger import logger
from ..metrics.correctness import calc_correctness
from . import stylex_builders
from .base import BaseTransformer

# import Java classes manually
Configuration = jp.JClass('org.example.Configuration')
Applicator = jp.JClass('org.example.controller.Applicator')
Extractor = jp.JClass('org.example.controller.Extractor')
StylerContainer = jp.JClass('org.example.controller.StylerContainer')
TokenAugmentor = jp.JClass('org.example.controller.TokenAugmentor')
GlobalInfo = jp.JClass('org.example.global.GlobalInfo')
ApplyException = jp.JClass('org.example.myException.ApplyException')
ExtractException = jp.JClass('org.example.myException.ExtractException')
MyParseTreeWalker = jp.JClass('org.example.parser.common.MyParseTreeWalker')
MyParserFactory = jp.JClass('org.example.parser.common.factory.MyParserFactory')
Spot = jp.JClass('org.example.parser.java.Spot')
SpotDetectorListener = jp.JClass('org.example.parser.java.SpotDetectorListener')
ProgramStyle = jp.JClass('org.example.style.ProgramStyle')
StyleFileIO = jp.JClass('org.example.style.StyleFileIO')
Stage = jp.JClass('org.example.styler.Stage')

with open('configs/settings.yaml', 'r') as f:
  config = yaml.safe_load(f)['transformer']

pict_path = os.getenv('PICT_PATH', 'pict')
if not pict_path or not shutil.which(pict_path):
  raise ValueError(f'PICT_PATH is not set or the pict executable is not found at {pict_path}.')


def set_global_info(func: Callable) -> Callable:
  def wrapper(self, lang: str, *args, **kwargs):
    # boilerplate to use StyleX
    GlobalInfo.setConf(Configuration())
    GlobalInfo.setLanguage(lang)
    return func(self, lang, *args, **kwargs)
  return wrapper


class StyleX(BaseTransformer):
  def transform(
      self,
      snippets: Sequence[Snippet],
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
    :return: a series of transformed snippets, each of which is corresponding to a variant sequence
    """
    style_file = os.getenv('STYLE_FILE')
    if not style_file:
      return self.span(lang, snippets, seed, ensure_correct)
    return [self.apply(lang, snippets, style_file)]

  @set_global_info
  def apply(
      self,
      lang: str,
      snippets: Sequence[Snippet],
      style_file: str,
  ) -> Sequence[Snippet | None]:
    styler_container = self._extract_from_file(lang, style_file)

    variants = [None] * len(snippets)
    for i, snippet in tqdm(enumerate(snippets), desc='Transforming', total=len(snippets), leave=False):
      try:
        variant = self._apply_styles(lang, snippet.code, styler_container)
      except Exception as e:
        logger.error(f'Error occurred for snippet {snippet.id}.\n{e}')
        variant = None
      if not variant:
        logger.warning(f'Failed to transform snippet {i} ({snippet.id}).')
        variants[i] = None
      else:
        variants[i] = snippet.replace(code=str(variant))
    return variants

  @set_global_info
  def span(
      self,
      lang: str,
      snippets: Sequence[Snippet],
      seed: int,
      ensure_correct: bool,
  ) -> Sequence[Sequence[Snippet | None]]:
    def worker(snippet_idx: int, seq_idx: int, snippet: Snippet, seq: Sequence[int]) -> Snippet | None:
      if seq_idx in snippet.args.get('transformed_seqs', set()):
        return None
      try:
        # TODO: fall back with self style?
        styler_container = self._build_stylers(seq)
        variant_code = self._apply_styles(lang, snippet.code, styler_container)
        if variant_code and ensure_correct:
          correctness = calc_correctness([snippet.replace(code=variant_code)], lang)
          if not math.isclose(correctness, 1.0):
            logger.warning(f'Correctness check failed: {correctness}')
            variant_code = None
      except Exception as e:
        logger.error(f'Error occurred while spanning:\n{e}')
        variant_code = None
      if not variant_code:
        logger.warning(f'Failed to transform snippet {snippet_idx} ({snippet.id}) with sequence {seq_idx} ({seq}).')
        return None
      logger.debug(f'Successfully transformed snippet {snippet_idx} ({snippet.id}).')
      return snippet.replace(code=str(variant_code))

    seqs = self._generate_seqs(seed)
    num_seq = len(seqs)
    corpus = [None] * len(snippets)
    for i, snippet in tqdm(enumerate(snippets), desc='Spanning', total=len(snippets), leave=False):
      with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        results = list(tqdm(executor.map(worker, [i] * num_seq, range(num_seq),
                                         [snippet] * num_seq, seqs),
                            desc=f'Spanning snippet {i}', total=num_seq, leave=False))
      corpus[i] = results
    logger.info(f'Spanned {len(corpus)} variant benchmarks.')
    return corpus

  def count_spots(
      self,
      lang: str,
      snippets: Sequence[Snippet],
  ) -> int:
    counts = [None] * len(snippets)
    for i, snippet in enumerate(snippets):
      spots = jp.java.util.HashMap()
      parser = MyParserFactory.createParser(lang)
      listener = SpotDetectorListener(spots, parser)
      tree = parser.parseFromString(snippet.code)
      walker = MyParseTreeWalker()
      walker.walk(listener, tree)
      counts[i] = spots.size()
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

  def _generate_seqs(
      self,
      seed: int,
  ) -> Sequence[Sequence[int]]:
    with open('configs/stylex_options.yaml', 'r') as f:
      option_config = yaml.safe_load(f)
    option_counts = []
    for style in chain(*[list(item.values())[0] for item in option_config]):
      option = list(style.values())[0]
      match option.get('option_type'):
        case 'bool':
          count = 2
        case 'enum':
          count = len(option.get('options', []))
        case 'number':
          count = option.get('length', 0)
        case _:
          raise TypeError(f'Unknown option type: {option.get("option_type")}')
      option_counts.append(count)

    with NamedTemporaryFile('w', encoding='utf-8', prefix='model', suffix='.txt', delete=False) as f:
      f.write('\n'.join([f'{i}: {",".join(map(str, range(count)))}' for i, count in enumerate(option_counts)]))
      f.flush()
    try:
      args = [pict_path, f.name, f'/r:{seed}']
      completed = subprocess.run(args, check=True, encoding='utf-8', stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
      logger.error(f'Error occurred while running pict.\n{e}')
      raise e
    os.remove(f.name)
    seqs = [[int(num) for num in line.split()] for line in completed.stdout.splitlines()[1:]]
    return seqs

  def _build_stylers(
      self,
      seq: Sequence[int],
  ) -> jp.JObject:
    with open('configs/stylex_options.yaml', 'r') as f:
      option_config = yaml.safe_load(f)
    option_dict = defaultdict(dict)
    idx = 0
    for item in option_config:
      styler_name = list(item.keys())[0]
      styles = item[styler_name]
      for i in range(len(styles)):
        style_name = list(styles[i].keys())[0]
        option_item = styles[i][style_name]
        match option_item.get('option_type'):
          case 'bool':
            option = 'true' if seq[idx] == 1 else 'false'
          case 'enum':
            option = option_item.get('options')[seq[idx]]
          case 'number':
            option = str(seq[idx])
          case _:
            raise TypeError(f'Unknown option type: {option_item.get("option_type")}')
        option_dict[styler_name][style_name] = option
        idx += 1

    container = StylerContainer()
    for styler in container.getStylers():
      if not styler.isEnable(Stage.APPLY):
        continue
      styler_name = styler.getClass().getSimpleName()
      try:
        options = option_dict[styler_name]
        builder = getattr(stylex_builders, f'{styler_name}Builder')
        style = builder.build(options)
        styler.setStyle(style)
      except Exception as e:
        logger.warning(f'Error occurred while building styler {styler_name}:\n{e}')
    return container
