from collections.abc import Sequence

from ... import Snippet
from ...logger import logger
from ..agents import BaseAgent, Prompt
from ..utils import work

SYSTEM_PROMPT = """
<task>
Given a code snippet, select the most probable option that describes the behavior while running.
</task>
<constraint>
Your output MUST only contain the letter (A, B, C, or D) corresponding to the correct answer without any explanation.
</constraint>
"""
USER_PROMPT = """
<code>```{lang}
{code}
```</code>
<choices>
{choices}
</choices>
"""


def answer_to_mcq(agent: BaseAgent, snippets: Sequence[Snippet | None], lang: str) -> Sequence[str | None]:
  def worker(i: int, snippet: Snippet) -> str | None:
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(lang=lang, code=snippet.code,
                                            choices='\n'.join([f'{letter}. {content}' for letter, content in zip('ABCD', snippet.args['choices'])])))
    response = agent.generate(prompt).strip()
    if not response:
      logger.warning(f'Empty answer for snippet {i} ({snippet.id}).')
      return None
    logger.debug(f'Answer for snippet {i}: {response}')
    return response[0] if response else None

  return work(worker=worker, snippets=snippets)

