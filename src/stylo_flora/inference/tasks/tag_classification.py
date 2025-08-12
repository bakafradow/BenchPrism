from collections.abc import Sequence

from ... import Snippet
from ...logger import logger
from ..agents import BaseAgent, Prompt
from ..utils import work

SYSTEM_PROMPT = """
<task>
Classify code into one or more categories from the following candidates.
</task>
<candidates>
2-sat,binary search,bitmasks,brute force,combinatorics,constructive algorithms,data structures,dfs and similar,divide and conquer,dp,dsu,expression parsing,fft,flows,games,geometry,graph matchings,graphs,greedy,implementation,interactive,math,matrices,meet-in-the-middle,number theory,probabilities,shortest paths,sortings,strings,trees,two pointers
</candidates>
<constraint>
Your output MUST only contain the exact list of categories separated by commas, not enclosed by any quotes or brackets, without any explanation.
</constraint>
"""
USER_PROMPT = """
<code>```{lang}
{code}
```</code>
"""


def tag(agent: BaseAgent, snippets: Sequence[Snippet], lang: str, with_desc: bool) -> Sequence[Sequence[str] | None]:
  def worker(i: int, snippet: Snippet) -> Sequence[str] | None:
    if not snippet or snippet.args.get('performed'):
      return None
    desc = f'\n<problem_description>{snippet.args["desc"]}</problem_description>\n' if with_desc else ''
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(lang=lang, code=snippet.code) + desc)
    response = agent.generate(prompt).strip()
    if not response:
      logger.warning(f'Empty tags for snippet {i} ({snippet.id}).')
      return None
    tags = [tag.strip() for tag in response.split(',')]
    logger.debug(f'Tags for snippet {i}: {tags}')
    return tags

  return work(worker=worker, snippets=snippets)
