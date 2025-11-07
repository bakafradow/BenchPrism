import json
import re
from collections.abc import MutableSequence as MSeq
from collections.abc import Sequence as Seq

from ... import IOTestCase, Snippet
from ...logger import logger
from ..agents import BaseAgent, Prompt
from ..utils import work

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


def generate_tests(agent: BaseAgent, snippets: Seq[Snippet | None], lang: str) -> MSeq[Seq[IOTestCase] | None]:
  def worker(i: int, snippet: Snippet) -> Seq[IOTestCase] | None:
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(
                        lang=lang,
                        desc=snippet.args['desc'],
                        input_spec=snippet.args['input_spec'],
                        output_spec=snippet.args['output_spec'],
                        sample_inputs=repr(snippet.args['sample_inputs']),
                        sample_outputs=repr(snippet.args['sample_outputs']),
                        code=snippet.code,
                        notes=snippet.args['notes'],
                    ))
    response = agent.generate(prompt)
    matched = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
    if not matched:
      logger.warning(f'Test cases of {snippet.id} not found.')
      logger.debug(response)
      return None
    json_str = matched.group(0)
    try:
      testcases = json.loads(json_str, strict=False)
      logger.debug(f'Snippet {i}:\n{json_str}')
      return [IOTestCase(input=testcase['input'][0] if isinstance(testcase['input'], list) \
                         else testcase['input'],
                         outputs=testcase['output'] if isinstance(testcase['output'], list) \
                         else [testcase['output']])
              for testcase in testcases]
    except Exception as e:
      logger.warning(f'Failed to parse test cases of {snippet.id}:\n{e}')
      logger.debug(response)
      return None

  return work(worker=worker, snippets=snippets)
