import re

from ... import Snippet
from ...logger import logger
from .base import BaseTask


class CodeTranslation(BaseTask):
  SYSTEM_PROMPT = """
  <task>
  Translate code from one language to another without changing its behavior.
  </task>
  <constraint>
  Your output MUST only contain the translated code WITHOUT any explanation, enclosed by triple back quotes with the language specified.
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

  def __init__(self, src_lang: str, dst_lang: str) -> None:
    super().__init__()
    self.src_lang = src_lang
    self.dst_lang = dst_lang

  def get_prompt(self, snippet: Snippet) -> tuple[str, str]:
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        src_lang=self.src_lang,
        dst_lang=self.dst_lang,
        code=snippet.data['code'],
    )

  def resolve_response(self, res: str) -> str | None:
    matched = re.search(r'```(?:\w+)?\n(.+)```', res, re.DOTALL)
    if not matched:
      logger.debug(res)
      return None
    return matched.group(1)
