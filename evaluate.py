import pandas as pd


def evaluate(snippets: pd.DataFrame, mutations: pd.DataFrame, dst_lang: str) -> None:
  """
  Evaluates the space spanned by the translated code relative to the original source code
  :param snippets: the translated original code snippets
  :param mutations: the translated mutated code snippets
  """
  print('Evaluating...')
