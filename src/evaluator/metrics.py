from typing import Sequence

from codebleu import calc_codebleu

from .. import Snippet


def average_codebleu_score(src: Sequence[Snippet], dst: Sequence[Snippet], lang: str) -> dict:
  """
  Calculate the average CodeBLEU score for the variants code.
  :param src: the source code snippets
  :param dst: the translated code snippets
  :param lang: the language of the code snippets
  :return: the average CodeBLEU score for each pair of snippets
  """
  if len(src) != len(dst):
    raise ValueError('The size of 2 snippet sequences should equal.')
  if lang not in ['java', 'cpp', 'python']:
    raise TypeError(f'Unsupported language: {lang}.')
  return calc_codebleu([snippet.code for snippet in src], [snippet.code for snippet in dst],
                       lang, weights=(.25, .25, .25, .25), tokenizer=None)
