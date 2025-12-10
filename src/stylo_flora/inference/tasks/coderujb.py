"""
Modified from CoderUJB repository (https://github.com/ZZR0/CoderUJB).
"""

import re

from ... import Snippet
from ...logger import logger
from .base import BaseTask

PATTERN_FUNC = re.compile(r'class\s+[\w\$]+[^\{]+\{\n(.+)\n\}', re.S)


class CodeRepairUJB(BaseTask):
  SYSTEM_PROMPT = ''

  USER_PROMPT = """{prefix}{buggy}
```
Your output MUST only contain the repaired function WITHOUT any explanation, enclosed by triple back quotes.
"""

  def __init__(self, lang: str) -> None:
    super().__init__()
    self.lang = lang

  def get_prompt(self, snippet: Snippet) -> tuple[str, str] | None:
    matched = PATTERN_FUNC.search(snippet.data['code'])
    if not matched:
      logger.warning(f'Failed to unwrap function for {snippet.id}.')
      return None
    buggy = matched.group(1)
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        prefix=snippet.data['prompt_prefix'],
        buggy=buggy,
    )

  def resolve_response(self, res: str) -> str | None:
    matched = re.search(r'```(?:\w+)?\n(.+)```', res, re.S)
    if not matched:
      logger.debug(res)
      return None
    return matched.group(1)
