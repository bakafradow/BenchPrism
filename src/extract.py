import logging
from typing import Sequence

from . import Snippet
from .utils import extract_field_from


def extract_source(dataset: str, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
  """
  Extracts source code from different datasets into unified data structure.
  :param dataset: dataset name
  :param src_lang: source language
  :param dst_lang: destination language
  :return: source code extracted from the dataset
  """
  logging.info(f'Extracting {src_lang} snippets from {dataset}...')
  match dataset:
    case 'HumanEvalX':
      task_ids = extract_field_from(dataset, src_lang, 'task_id')
      declarations = extract_field_from(dataset, src_lang, 'declaration')
      canonical_solutions = extract_field_from(dataset, src_lang, 'canonical_solution')
      sources = [declaration + '\n' + canonical_solution for declaration, canonical_solution in zip(declarations, canonical_solutions)]
      dst_declarations = extract_field_from(dataset, dst_lang, 'declaration')
      return list(map(lambda pair: Snippet(*pair), zip(task_ids, sources, dst_declarations)))
    case 'xCodeEval':
      src_uids = extract_field_from(dataset, src_lang, 'src_uid')
      sources = extract_field_from(dataset, src_lang, 'source_code')
      return list(map(lambda pair: Snippet(*pair), zip(src_uids, sources)))
    case 'XLCoST':
      sources = extract_field_from(dataset, src_lang, 'code')
      return list(map(lambda pair: Snippet(*pair), enumerate(sources)))
    case 'CodeXGLUE':
      ids = extract_field_from(dataset, src_lang, 'id')
      sources = extract_field_from(dataset, src_lang, '')
      return list(map(lambda pair: Snippet(*pair), zip(ids, sources)))
    case 'G-TransEval':
      ids = extract_field_from(dataset, src_lang, 'id')
      sources = extract_field_from(dataset, src_lang, 'content')
      return list(map(lambda pair: Snippet(*pair), zip(ids, sources)))
    case _:
      raise TypeError(f'Unknown dataset: {dataset}.')
