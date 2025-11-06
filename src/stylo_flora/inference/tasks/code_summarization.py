import reprlib
from collections.abc import MutableSequence as MSeq
from collections.abc import Sequence as Seq

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


def summarize(agent: BaseAgent, snippets: Seq[Snippet | None], lang: str) -> MSeq[str | None]:
  def worker(i: int, snippet: Snippet) -> str | None:
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(lang=lang, code=snippet.code))
    response = agent.generate(prompt).strip()
    logger.debug(f'Summary for snippet {i}: {reprlib.repr(response)}')
    if not response:
      logger.warning(f'Empty summary for snippet {i} ({snippet.id}).')
      return None
    return response

  return work(worker=worker, snippets=snippets)
