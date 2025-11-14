from ... import Snippet
from .base import BaseTask


class MCQAnswering(BaseTask):
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

  def __init__(self, lang: str):
    super().__init__()
    self.lang = lang

  def get_prompt(self, snippet: Snippet) -> tuple[str, str]:
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        lang=self.lang,
        code=snippet.data['code'],
        choices='\n'.join([f'{letter}. {content}' for letter, content in zip('ABCD', snippet.data['choices'])])
    )

  def resolve_response(self, res: str) -> str | None:
    res = res.strip()
    return res[0] if res else None
