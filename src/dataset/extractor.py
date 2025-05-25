from typing import Collection, Sequence

from datasets import load_dataset

from .. import Snippet
from ..logger import logger

def _check_lang_support(lang: str, supported_langs: Collection[str]):
  if lang not in supported_langs:
    raise TypeError(f'{lang} is not supported in current dataset.')


def extract_field_from(dataset: str, lang: str, column: str) -> Sequence[str]:
  """
  Extracts specific field from a dataset.
  :param dataset: dataset name
  :param lang: one of the supported languages in the dataset
  :param column: column name
  :return: the field extracted from the dataset
  """
  match dataset:
    case 'HumanEvalX':
      _check_lang_support(lang, ['python', 'cpp', 'go', 'java', 'js'])
      ds = load_dataset('THUDM/humaneval-x', lang, trust_remote_code=True)
      return [row[column] for row in ds['test']]
    case 'xCodeEval':
      _check_lang_support(lang, ['c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust'])
      lang_to_name = {
        'c': 'C',
        'cpp': 'C++',
        'cs': 'C#',
        'go': 'Go',
        'java': 'Java',
        'js': 'Javascript',
        'kotlin': 'Kotlin',
        'php': 'PHP',
        'python': 'Python',
        'ruby': 'Ruby',
        'rust': 'Rust',
      }
      ds = load_dataset('json', data_dir='data/xCodeEval/code_translation')  # there's an issue when loading from HF
      lang_name = lang_to_name[lang]
      ds = ds.filter(lambda row: row['lang_cluster'] == lang_name)
      return ds['test'][column]
    case 'XLCoST':
      _check_lang_support(lang, ['c', 'cs', 'cpp', 'java', 'js', 'php', 'python'])
      lang_to_name = {
        'c': 'C',
        'cs': 'Csharp',
        'cpp': 'C++',
        'java': 'Java',
        'js': 'Javascript',
        'php': 'PHP',
        'python': 'Python',
      }
      lang_name = lang_to_name[lang]
      ds = load_dataset('codeparrot/xlcost-text-to-code', f'{lang_name}-program-level')
      return ds['train'][column]
    case 'CodeXGLUE':
      _check_lang_support(lang, ['cs', 'java'])
      ds = load_dataset('google/code_x_glue_cc_code_to_code_trans', trust_remote_code=True)
      return ds['train'][lang]  # returns the same whatever the column is
    case 'G-TransEval':
      _check_lang_support(lang, ['cpp', 'java', 'python'])
      ds = load_dataset(f'xin1997/g-transeval-{lang}_all_only_input', trust_remote_code=True)
      return ds['train'][column]
    case _:
      raise TypeError(f'Unknown dataset: {dataset}.')


def extract_source(dataset: str, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
  """
  Extracts source code from different datasets into unified data structure.
  :param dataset: dataset name
  :param src_lang: source language
  :param dst_lang: destination language
  :return: source code extracted from the dataset
  """
  logger.info(f'Extracting {src_lang} snippets from {dataset}...')
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
