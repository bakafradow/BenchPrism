from ... import Snippet
from .base import BaseTask


class TagClassification(BaseTask):
  SYSTEM_PROMPT = """<task>
Classify code into one or more categories from the following candidates.
</task>
<candidates>
2-sat,binary search,bitmasks,brute force,combinatorics,constructive algorithms,data structures,dfs and similar,divide and conquer,dp,dsu,expression parsing,fft,flows,games,geometry,graph matchings,graphs,greedy,implementation,interactive,math,matrices,meet-in-the-middle,number theory,probabilities,shortest paths,sortings,strings,trees,two pointers
</candidates>
<constraint>
Your output MUST only contain the exact list of categories separated by commas, not enclosed by any quotes or brackets, without any explanation.
</constraint>
"""

  USER_PROMPT = """<code>```{lang}
  {code}
  ```</code>
  """

  def __init__(self, lang: str, with_desc: bool):
    super().__init__()
    self.lang = lang
    self.with_desc = with_desc

  def get_prompt(self, snippet: Snippet) -> tuple[str, str] | None:
    desc = f'\n<problem_description>{snippet.data["desc"]}</problem_description>\n' if self.with_desc else ''
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        lang=self.lang,
        code=snippet.data['code'],
    ) + desc

  def resolve_response(self, res: str) -> list[str] | None:
    if not res:
      return None
    tags = [tag.strip() for tag in res.split(',')]
    return tags
