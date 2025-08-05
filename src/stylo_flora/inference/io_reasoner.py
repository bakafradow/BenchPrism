"""
Modified from CRUXEval-X repository (https://github.com/CRUXEVAL-X/cruxeval-x).
"""

import re
from collections.abc import Sequence

from .. import Snippet
from ..logger import logger
from .agents import BaseAgent, Prompt
from .utils import work

SYSTEM_PROMPT = """
<task>
Based on the given code, which may contain errors, reason out the "????" in the assertion statement to make it compilable and correct.
</task>
<constraint>
Your output MUST only contain the exact expression that should replace the "????" in the assertion statement without any explanations, enclosed by triple back quotes.
</constraint>
"""
USER_PROMPT = """
<code>```{lang}
{code}
```</code>
"""


def _reason(agent: BaseAgent, snippets: Sequence[Snippet], lang: str) -> Sequence[Sequence[Snippet]]:
  def worker(i: int, snippet: Snippet) -> Sequence[str] | None:
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(lang=lang, code=snippet.code))
    response = agent.generate(prompt)
    matched = re.search(r'```(?:\w+)?\n(.+)```', response, re.DOTALL)
    if not matched:
      logger.warning(f'Content of {snippet.id} not found.')
      logger.debug(response)
      return None
    logger.debug(f'Content for snippet {i}: {response}')
    return snippet._replace(code=snippet.code.replace('????', matched.group(1).strip()))

  return work(worker=worker, snippets=snippets)


def _add_main(lang: str, code: str, main_part: str) -> str:
  match lang:
    case 'java':
      # remove the last 2 right braces and append the main part
      return re.sub(r'\s*}\s*}\s*$', '\n', code) + main_part
    case _:
      raise ValueError(f'Unsupported language: {lang}')


def reason_input(agent: BaseAgent, snippets: Sequence[Snippet], lang: str) -> Sequence[Sequence[Snippet]]:
  snippets_without_input = [snippet._replace(code=_add_main(lang, snippet.code, snippet.args['input_reasoning'])) for snippet in snippets]
  return _reason(agent, snippets_without_input, lang)


def reason_output(agent: BaseAgent, snippets: Sequence[Snippet], lang: str) -> Sequence[Sequence[Snippet]]:
  snippets_without_output = [snippet._replace(code=_add_main(lang, snippet.code, snippet.args['output_reasoning'])) for snippet in snippets]
  return _reason(agent, snippets_without_output, lang)
