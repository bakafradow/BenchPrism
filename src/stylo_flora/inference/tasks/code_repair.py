import re

from ... import Snippet
from ...logger import logger
from .base import BaseTask


class CodeRepair(BaseTask):
  SYSTEM_PROMPT = """<task>
Repair buggy code to make it work correctly.
</task>
<constraint>
Your output MUST only contain the repaired code WITHOUT any explanation, enclosed by triple back quotes and the language specified.
</constraint>
"""

  USER_PROMPT = """<language>{lang}</language>
<problem_description>{desc}</problem_description>
<input_specification>{input_spec}</input_specification>
<output_specification>{output_spec}</output_specification>
<sample_inputs>```
{sample_inputs}
```</sample_inputs>
<sample_outputs>```
{sample_outputs}
```</sample_outputs>
<buggy_code>```{lang}
{code}
```</buggy_code>
<error_message>{msg}</error_message>
"""

  def __init__(self, lang: str) -> None:
    super().__init__()
    self.lang = lang

  def get_prompt(self, snippet: Snippet) -> tuple[str, str] | None:
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        lang=self.lang,
        desc=snippet.data['desc'],
        input_spec=snippet.data['input_spec'],
        output_spec=snippet.data['output_spec'],
        sample_inputs=repr(snippet.data['sample_inputs']),
        sample_outputs=repr(snippet.data['sample_outputs']),
        code=snippet.data['code'],
        msg=snippet.data.get('error_msg', ''),
    )
  
  def resolve_response(self, res: str) -> str | None:
    matched = re.search(r'```(?:\w+)?\n(.+)```', res, re.S)
    if not matched:
      logger.debug(res)
      return None
    return matched.group(1)
