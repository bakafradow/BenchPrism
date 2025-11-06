import re
from collections.abc import MutableSequence as MSeq
from collections.abc import Sequence as Seq

from ... import Snippet
from ...logger import logger
from ..agents import BaseAgent, Prompt
from ..utils import work

SYSTEM_PROMPT = """
<task>
Repair buggy code to make it work correctly.
</task>
<constraint>
Your output MUST only contain the repaired code WITHOUT any explanation, enclosed by triple back quotes and the language specified.
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
<buggy_code>```{lang}
{code}
```</buggy_code>
<error_message>{msg}</error_message>
"""


def repair(agent: BaseAgent, snippets: Seq[Snippet | None], lang: str) -> MSeq[Snippet | None]:
  def worker(i: int, snippet: Snippet) -> Snippet | None:
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(
                      lang=lang,
                      desc=snippet.args['desc'],
                      input_spec=snippet.args['input_spec'],
                      output_spec=snippet.args['output_spec'],
                      sample_inputs=repr(snippet.args['sample_inputs']),
                      sample_outputs=repr(snippet.args['sample_outputs']),
                      code=snippet.code,
                      msg=snippet.args.get('error_msg', 'No error message provided.'),
                    ))
    response = agent.generate(prompt)
    matched = re.search(r'```(?:\w+)?\n(.+)```', response, re.DOTALL)
    if not matched:
      logger.warning(f'Repaired version of {snippet.id} not found.')
      logger.debug(response)
      return None
    logger.debug(f'Snippet {i}:\n{matched.group(1)}')
    return snippet.replace(code=matched.group(1))

  return work(worker=worker, snippets=snippets)

