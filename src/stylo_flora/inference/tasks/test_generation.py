import json
import re
from collections.abc import Sequence as Seq

from ... import IOTestCase, Snippet
from ...logger import logger
from .base import BaseTask


class TestGeneration(BaseTask):
  SYSTEM_PROMPT = """
  <task>
  Provide exactly 5 test cases for a given problem along with its solution.
  </task>
  <constraint>
  1. Each test case contains a string for both input and output.
  2. The solution code successfully processes the test case's input without errors and the outcome aligns with the test case's output.
  3. All test cases are simple and achieve optimal branch and line coverage.
  4. Your response MUST only contain a string in the following JSON format:
  [{{"input": input string, "output": output string}}]
  </constraint>
  """

  USER_PROMPT = """
  <language>{lang}</language>
  <problem_description>{desc}</problem_description>
  <input_specification>{input_spec}</input_specification>
  <output_specification>{output_spec}</output_specification>
  <sample_inputs>```
  {sample_inputs}
  ```</sample_inputs>
  <sample_outputs>```
  {sample_outputs}
  ```</sample_outputs>
  <code>```{lang}
  {code}
  ```</code>
  <notes>{notes}</notes>
  """

  def __init__(self, lang: str):
    super().__init__()
    self.lang = lang

  def get_prompt(self, snippet: Snippet) -> tuple[str, str]:
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        lang=self.lang,
        desc=snippet.data['desc'],
        input_spec=snippet.data['input_spec'],
        output_spec=snippet.data['output_spec'],
        sample_inputs=repr(snippet.data['sample_inputs']),
        sample_outputs=repr(snippet.data['sample_outputs']),
        code=snippet.data['code'],
        notes=snippet.data['notes'],
    )

  def resolve_response(self, res: str) -> Seq[IOTestCase] | None:
    matched = re.search(r'\[\s*\{.*?\}\s*\]', res, re.DOTALL)
    if not matched:
      logger.debug(res)
      return None
    json_str = matched.group(0)
    try:
      tc_list = json.loads(json_str, strict=False)
      logger.debug(f'Test cases:\n{json_str}')
      return [IOTestCase.from_dict(tc) for tc in tc_list]
    except Exception as e:
      logger.warning(f'Failed to parse test cases:\n{e}')
      logger.debug(res)
      return None
