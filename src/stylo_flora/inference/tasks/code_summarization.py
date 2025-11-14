from ... import Snippet
from .base import BaseTask


class CodeSummarization(BaseTask):
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

  def __init__(self, lang: str) -> None:
    super().__init__()
    self.lang = lang

  def get_prompt(self, snippet: Snippet) -> tuple[str, str]:
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        lang=self.lang,
        code=snippet.data['code'],
    )
  
  def resolve_response(self, res: str) -> str | None:
    return res.strip() or None
