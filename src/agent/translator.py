import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import yaml
from tqdm import tqdm

from .. import Snippet
from ..logger import logger
from .base import BaseAgent, Prompt

with open('settings.yml') as f:
  config = yaml.safe_load(f)['agent']


SYSTEM_PROMPT = """
<task>
Translate code from one language to another without changing its behavior.
</task>
<constraint>
Your output MUST only contain the translated code WITHOUT any explanations, enclosed by triple back quotes.
</constraint>
"""
USER_PROMPT = """
<source_language>{src_lang}</source_language>
<target_language>{dst_lang}</target_language>
<source_code>```{src_lang}\n{code}```</source_code>
"""


def translate(translator: BaseAgent, snippets: Sequence[Snippet], src_lang: str, dst_lang: str) -> Sequence[Snippet]:
  """
  Translates snippets in code set with code translation model.
  :param translator: the translation model
  :param snippets: the snippets to be translated
  :param dataset: dataset name
  :param src_lang: source language
  :param dst_lang: destination language
  :return: a sequence of translated code
  """
  logger.info(f'Translating from {src_lang} to {dst_lang}...')

  def worker(i: int, snippet: Snippet) -> Snippet | None:
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(src_lang=src_lang, dst_lang=dst_lang, code=snippet.code))
    response = translator.generate(prompt)
    matched = re.search(r'```(?:\w+)?\n(.+)```', response, re.DOTALL)
    if not matched:
      logger.warning(f'Translation of {snippet.id} not found.')
      logger.verbose(response)
      return None
    logger.verbose(f'Snippet {i}:\n{matched.group(1)}')
    return snippet._replace(code=matched.group(1))
  max_workers = max(1, config['max_workers'])
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    return tuple(tqdm(executor.map(worker, range(len(snippets)), snippets),
                      desc='Translating snippets', total=len(snippets), leave=False))
