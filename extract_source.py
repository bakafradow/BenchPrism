import pandas as pd
from datasets import load_dataset
from pprint import pprint


def _check_lang_support(src_lang: str, supported_langs: list[str]):
  if src_lang not in supported_langs:
    raise ValueError(f'{src_lang} is not supported in current dataset.')


def _to_dataframe(ids: list[str | int], snippets: [list[str]]) -> pd.DataFrame:
  return pd.DataFrame({'id': ids, 'snippet': snippets}, dtype=str)


def extract(dataset: str, src_lang: str) -> pd.DataFrame:
  """
  Extracts source code from different datasets into unified format.
  :param dataset: dataset name
  :param src_lang: source language
  :return: source code extracted from the dataset
  """
  print(f'\nExtracting source code from {dataset} dataset...')
  match dataset:
    case 'HumanEvalX':
      supported_langs = ['python', 'cpp', 'go', 'java', 'js']
      _check_lang_support(src_lang, supported_langs)
      ds = load_dataset('THUDM/humaneval-x', src_lang, trust_remote_code=True)
      snippets = [f'{row["declaration"]}{row["canonical_solution"]}' for row in ds['test']]
      return _to_dataframe(ds['test']['task_id'], snippets)
    case 'xCodeEval':
      supported_langs = ['c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust']
      _check_lang_support(src_lang, supported_langs)
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
      ds = ds.filter(lambda row: row['lang_cluster'] == lang_to_name[src_lang])
      return _to_dataframe(ds['train']['src_uid'], ds['train']['source_code'])
    case 'XLCoST':
      supported_langs = ['c', 'cs', 'cpp', 'java', 'js', 'php', 'python']
      _check_lang_support(src_lang, supported_langs)
      lang_to_name = {
        'c': 'C',
        'cs': 'Csharp',
        'cpp': 'C++',
        'java': 'Java',
        'js': 'Javascript',
        'php': 'PHP',
        'python': 'Python',
      }
      ds = load_dataset('codeparrot/xlcost-text-to-code', f'{lang_to_name[src_lang]}-program-level')
      return _to_dataframe(list(range(ds['train'].shape[0])), ds['train']['code'])
    case 'CodeXGLUE':
      supported_langs = ['cs', 'java']
      _check_lang_support(src_lang, supported_langs)
      ds = load_dataset('google/code_x_glue_cc_code_to_code_trans', trust_remote_code=True)
      return _to_dataframe(ds['train']['id'], ds['train']['java'])
    case 'G-TransEval':
      supported_langs = ['cpp', 'java', 'python']
      _check_lang_support(src_lang, supported_langs)
      ds = load_dataset(f'xin1997/g-transeval-{src_lang}_all_only_input', trust_remote_code=True)
      return _to_dataframe(ds['train']['id'], ds['train']['content'])
    case _:
      raise ValueError(f'Unknown dataset: {dataset}.')
