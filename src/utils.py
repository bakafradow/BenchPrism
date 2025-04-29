import argparse
import errno
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Collection, NamedTuple, Sequence

import yaml
from codebleu import calc_codebleu
from datasets import load_dataset

from . import Snippet

### CLI arguments

class Arguments(NamedTuple):
  datasets: list[str]
  model: str
  src_lang: str
  dst_lang: str
  gpu_id: int = -1
  num_snippets: int = -1


def parse_args() -> Arguments:
  default_datasets = ['HumanEvalX', 'xCodeEval', 'XLCoST', 'CodeXGLUE', 'G-TransEval', 'CodeNet']
  default_src_lang = 'java'
  default_dst_lang = 'cpp'
  parser = argparse.ArgumentParser(description='Code translation evaluation tool.'
                                               'All the datasets are evaluated by default.')
  parser.add_argument('-d', '--dataset', nargs=1, type=str,
                      choices=default_datasets,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str,
                      required=True,
                      help='Specify the model to use.')
  parser.add_argument('--src-lang', default=default_src_lang, type=str,
                      choices=['java', 'cpp'],
                      help=f'Specify the source language, {default_src_lang} by default.')
  parser.add_argument('--dst-lang', default=default_dst_lang,type=str,
                      choices=['c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust'],
                      help=f'Specify the destination language, {default_dst_lang} by default.')
  parser.add_argument('-i', '--gpu-id', type=int, default=-1,
                      help='Specify the GPU to use.')
  parser.add_argument('-n', '--num-snippets', type=int, default=-1,
                      help='Limit the number of snippets to test.')
  args = parser.parse_args()
  return Arguments(
    datasets=args.dataset if args.dataset else default_datasets,
    model=args.model,
    src_lang=args.src_lang,
    dst_lang=args.dst_lang,
    gpu_id=args.gpu_id,
    num_snippets=args.num_snippets,
  )


### Logging configuration

class ColorFormatter(logging.Formatter):
  COLORS = {
    'VERBOSE': '\033[38;5;52m',
    'INFO': '\033[38;5;91m',
    'WARNING': '\033[38;5;202m',
    'ERROR': '\033[38;5;161m',
    'CRITICAL': '\033[38;5;76m',
    'ENDC': '\033[0m',
  }

  def format(self, record):
    return f'{self.COLORS[record.levelname]}{super().format(record)}{self.COLORS["ENDC"]}'


with open('settings.yml') as f:
  logger_config = yaml.safe_load(f)['logger']


VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, 'VERBOSE')


def verbose(self, message, *args, **kwargs):
  if self.isEnabledFor(VERBOSE_LEVEL):
    self._log(VERBOSE_LEVEL, message, args, **kwargs)


logging.Logger.verbose = verbose

logger = logging.getLogger()
if logger_config['verbose']:
  logger.setLevel(VERBOSE_LEVEL)
else:
  logger.setLevel(logging.INFO)

pattern = '%(asctime)s - %(levelname)s - %(message)s'

log_path = os.getenv('LOG_FILE')
if not os.path.exists(os.path.dirname(log_path)):
  try:
    os.makedirs(os.path.dirname(log_path))
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
file_handler = RotatingFileHandler(log_path, mode='a', maxBytes=logger_config['max_bytes'], backupCount=logger_config['backup_count'])
file_handler.setFormatter(logging.Formatter(pattern))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(ColorFormatter(pattern))
logger.addHandler(stream_handler)


### Dataset

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


### Code similarity

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
  match lang:
    case 'java' | 'cpp' | 'python':
      pass  # do nothing
    case _:
      raise TypeError(f'Unsupported language: {lang}.')
  return calc_codebleu([snippet.code for snippet in src], [snippet.code for snippet in dst],
                       lang, weights=(.25, .25, .25, .25), tokenizer=None)
