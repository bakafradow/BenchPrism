import reprlib
from collections.abc import Sequence

from ... import Snippet
from ...logger import logger
from ..agents import BaseAgent, Prompt
from ..utils import work

SYSTEM_PROMPT = """
<task>
Summarize the code snippet briefly.
</task>
<constraint>
Your output MUST only contain the exact summary of the code snippet without any explanation.
</constraint>
"""
USER_PROMPT = """
<code>```{lang}
{code}
```</code>
"""


def summarize(agent: BaseAgent, snippets: Sequence[Snippet], lang: str) -> Sequence[Sequence[str]]:
  def worker(i: int, snippet: Snippet) -> Sequence[str] | None:
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(lang=lang, code=snippet.code))
    response = agent.generate(prompt)
    logger.debug(f'Summary for snippet {i}: {reprlib.repr(response)}')
    return response.strip()

  return work(worker=worker, snippets=snippets)
