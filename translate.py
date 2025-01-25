from typing import Sequence

import pandas as pd

from snippet import Snippet, SnippetSequence


def _load_model(model: str):
  pass


def translate_with_model(snippets: SnippetSequence, model: str, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
  """
  Translates snippets in code set with code translation model.
  :param snippets: the snippets to be translated
  :param model: name of the code translation model
  :param src_lang: source language
  :param dst_lang: destination language
  :return: a set of translated code
  """
  # TODO:
  #  1. load the model
  #  2. for each code snippet in the set, determine prompt and translate it
  #  3. return the translated code set
  print(f'Translating from {src_lang} to {dst_lang} with {model}...')
  model = _load_model(model)
  return pd.DataFrame()
