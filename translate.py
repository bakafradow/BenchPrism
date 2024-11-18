import pandas as pd
from typing import List


def translate_with_model(code_set: List[pd.DataFrame], model: str, src_lang: str, dst_lang: str) -> List[pd.DataFrame]:
  """
  Translates snippets in code set with code translation model.
  :param code_set: the set code to be translated
  :param model: name of the code translation model
  :param src_lang: source language
  :param dst_lang: destination language
  :return: a set of translated code
  """
  # TODO:
  #  1. load the model
  #  2. for each code snippet in the set, determine prompt and translate it
  #  3. return the translated code set
  pass
