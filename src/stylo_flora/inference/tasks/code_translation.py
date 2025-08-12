import re
from collections.abc import Sequence

from ... import Snippet
from ...logger import logger
from ..agents import BaseAgent, Prompt
from ..utils import work

SYSTEM_PROMPT = """
<task>
Translate code from one language to another without changing its behavior.
</task>
<constraint>
Your output MUST only contain the translated code WITHOUT any explanation, enclosed by triple back quotes.
Assertion statements, if exist, should also be considered.
Apply camel case in Java; apply snake case in C++ and Python.
</constraint>
"""
USER_PROMPT = """
<source_language>{src_lang}</source_language>
<target_language>{dst_lang}</target_language>
<source_code>```{src_lang}
{code}
```</source_code>
"""


def translate(agent: BaseAgent, snippets: Sequence[Snippet | None], src_lang: str, dst_lang: str) -> Sequence[Snippet | None]:
  """
  Translates snippets in code set with code translation model.
  :param translator: the translation model
  :param snippets: the snippets to be translated
  :param dataset: dataset name
  :param src_lang: source language
  :param dst_lang: destination language
  :return: a sequence of translated code
  """
  def worker(i: int, snippet: Snippet | None) -> Snippet | None:
    if not snippet or snippet.args.get('performed'):
      return None
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(src_lang=src_lang, dst_lang=dst_lang, code=snippet.code))
    response = agent.generate(prompt)
    matched = re.search(r'```(?:\w+)?\n(.+)```', response, re.DOTALL)
    if not matched:
      logger.warning(f'Translation of {snippet.id} not found.')
      logger.debug(response)
      return None
    logger.debug(f'Snippet {i}:\n{matched.group(1)}')
    return snippet._replace(code=matched.group(1))

  return work(worker=worker, snippets=snippets)
