from typing import Sequence

from . import Snippet


def evaluate(snippets: Sequence[Snippet], mutations: Sequence[Snippet], dst_lang: str) -> None:
  """
  Evaluates the space spanned by the translated code relative to the original source code
  :param snippets: the translated original code snippets
  :param mutations: the translated mutated code snippets
  :param dst_lang: destination language
  """
  print('Evaluating...')
