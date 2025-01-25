from typing import Sequence

import pandas as pd

from snippet import Snippet, SnippetSequence


def evaluate(snippets: SnippetSequence, mutations: SnippetSequence, dst_lang: str) -> None:
  """
  Evaluates the space spanned by the translated code relative to the original source code
  :param snippets: the translated original code snippets
  :param mutations: the translated mutated code snippets
  :param dst_lang: destination language
  """
  print('Evaluating...')
