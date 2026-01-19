"""
Modified from TestBench repository (https://github.com/iSEngLab/TestBench).
"""

import re

from ... import Snippet
from ...logger import logger
from .base import BaseTask

PATTERN_FUNC = re.compile(r'class\s+[\w\$]+[^\{]+\{\n(.+)\n\}', re.S)


class TestGenerationTB(BaseTask):
  SYSTEM_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request."""

  USER_PROMPT = """### Instruction:
Write a unit test for the following Java Source Code with junit. the Context information is given.
    
Unit test has been finished partially. Please complete the section contains <FILL> tag and output the whole test case, enclosed by triple back quotes.


### JAVA Source Code:
{source_code}
    

### Context:
{context}
    

### JUNIT Test case:
{test_info}
    

### Response:"""

  def __init__(self, lang: str):
    super().__init__()
    self.lang = lang

  def get_prompt(self, snippet: Snippet) -> tuple[str, str] | None:
    matched = PATTERN_FUNC.search(snippet.data['code'])
    if not matched:
      logger.warning(f'Failed to unwrap function for {snippet.id}.')
      return None
    source_code = matched.group(1)
    test_info = f"""
package {snippet.data['package']};

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

public class {snippet.data['class_name']}Test {{
    @Test
    public void {snippet.data['method_name']}Test() {{
        <FILL>
    }}
}}
"""
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        source_code=source_code,
        context=snippet.data['simple_context'],
        test_info=test_info,
    )

  def resolve_response(self, res: str) -> str | None:
    matched = re.search(r'```(?:\w+)?\n(.+)```', res, re.S)
    if not matched:
      logger.debug(res)
      return None
    return matched.group(1)
