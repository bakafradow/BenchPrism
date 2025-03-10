from typing import Collection, Sequence

from datasets import load_dataset

from . import Snippet


def _check_lang_support(src_lang: str, dst_lang: str, supported_langs: Collection[str]):
  if src_lang not in supported_langs:
    raise ValueError(f'{src_lang} is not supported in current dataset.')
  if dst_lang not in supported_langs:
    raise ValueError(f'{dst_lang} is not supported in current dataset.')


def extract_source(dataset: str, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
  """
  Extracts source code from different datasets into unified data structure.
  :param dataset: dataset name
  :param src_lang: source language
  :param dst_lang: destination language
  :return: source code extracted from the dataset
  """
  print(f'Extracting {src_lang} snippets from {dataset}...')
  match dataset:
    case 'HumanEvalX':
      _check_lang_support(src_lang, dst_lang, ['python', 'cpp', 'go', 'java', 'js'])
      ds = load_dataset('THUDM/humaneval-x', src_lang, trust_remote_code=True)
      snippets = [f'{row["declaration"]}{row["canonical_solution"]}' for row in ds['test']]
      return list(map(lambda pair: Snippet(*pair), zip(ds['test']['task_id'], snippets)))
    case 'xCodeEval':
      _check_lang_support(src_lang, dst_lang, ['c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust'])
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
      lang_name = lang_to_name[src_lang]
      ds = ds.filter(lambda row: row['lang_cluster'] == lang_name)
      return list(map(lambda pair: Snippet(*pair), zip(ds['train']['src_uid'], ds['train']['source_code'])))
    case 'XLCoST':
      _check_lang_support(src_lang, dst_lang, ['c', 'cs', 'cpp', 'java', 'js', 'php', 'python'])
      lang_to_name = {
        'c': 'C',
        'cs': 'Csharp',
        'cpp': 'C++',
        'java': 'Java',
        'js': 'Javascript',
        'php': 'PHP',
        'python': 'Python',
      }
      lang_name = lang_to_name[src_lang]
      ds = load_dataset('codeparrot/xlcost-text-to-code', f'{lang_name}-program-level')
      return list(map(lambda pair: Snippet(*pair), enumerate(ds['train']['code'])))
    case 'CodeXGLUE':
      _check_lang_support(src_lang, dst_lang, ['cs', 'java'])
      ds = load_dataset('google/code_x_glue_cc_code_to_code_trans', trust_remote_code=True)
      return list(map(lambda pair: Snippet(*pair), zip(ds['train']['id'], ds['train'][src_lang])))
    case 'G-TransEval':
      _check_lang_support(src_lang, dst_lang, ['cpp', 'java', 'python'])
      ds = load_dataset(f'xin1997/g-transeval-{src_lang}_all_only_input', trust_remote_code=True)
      return list(map(lambda pair: Snippet(*pair), zip(ds['train']['id'], ds['train']['content'])))
    case _:
      raise ValueError(f'Unknown dataset: {dataset}.')
