import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jsonlines
from datasets import load_dataset

from . import Snippet, TestBatch


def check_lang_support(func: Callable):
  def wrapper(self, lang, *args, **kwargs):
    if lang not in self.supported_langs:
      raise TypeError(f'{lang} is not supported in current benchmark. Supported languages: {self.supported_langs}')
    return func(self, lang, *args, **kwargs)
  return wrapper


@dataclass
class BaseBenchmark(ABC):
  """
  Abstract base class for benchmarks.
  """

  _supported_langs: frozenset[str] = field(default_factory=frozenset)
  """Supported languages in the dataset to evaluate."""
  _lang_to_name: dict[str, str] = field(default_factory=dict)
  """Mapping language name from unified one to the one in dataset."""

  @property
  def supported_langs(self) -> frozenset[str]:
    return self._supported_langs

  @supported_langs.setter
  def supported_langs(self, langs: frozenset[str]):
    self._supported_langs = langs

  @abstractmethod
  def load_source(self, lang: str) -> Sequence[Snippet]:
    """
    Loads the source code snippets for evaluation.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets
    """
    pass

  @abstractmethod
  def load_tests(self, ids: Iterable[str]) -> Sequence[TestBatch]:
    """
    Loads the test cases for the source code snippets.

    :param ids: the ids of the source code snippets
    :return: a sequence of test cases
    """
    pass


@dataclass
class HumanEvalX(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'python', 'cpp', 'go', 'java', 'js',
  ]))

  @check_lang_support
  def _load(self, lang: str, column: str) -> Sequence[str]:
    ds = load_dataset('THUDM/humaneval-x', lang, trust_remote_code=True)
    return tuple(row[column] for row in ds['test'])

  def load_source(self, lang: str) -> Sequence[Snippet]:
    task_ids = self._load(lang, 'task_id')
    declarations = self._load(lang, 'declaration')
    bodies = self._load(lang, 'canonical_solution')
    entries = self._load(lang, 'test')
    sources = (f'{declaration}\n{body}\n{entry}' for declaration, body, entry in zip(declarations, bodies, entries))
    return tuple(Snippet(id=task_id, code=source) for task_id, source in zip(task_ids, sources))

  def load_tests(self, ids: Iterable[str]) -> Sequence[TestBatch]:
    return tuple((('', ('',)),) for _ in ids)  # HumanEvalX evaluates correctness with assertions


@dataclass
class XCodeEval(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust',
  ]))
  _lang_to_name: dict[str, str] = field(default_factory=lambda: {
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
  })

  @check_lang_support
  def _load(self, lang, column) -> Sequence[str]:
    ds = load_dataset('json', data_dir='data/xCodeEval/code_translation')  # there's an issue when loading from HF
    lang_name = self._lang_to_name[lang]
    ds = ds.filter(lambda row: row['lang_cluster'] == lang_name)
    return ds['test'][column]

  def load_source(self, lang: str) -> Sequence[Snippet]:
    src_uids = self._load(lang, 'src_uid')
    sources = self._load(lang, 'source_code')
    return tuple(Snippet(id=src_uid, code=source) for src_uid, source in zip(src_uids, sources))

  def load_tests(self, ids: Iterable[str]) -> Sequence[TestBatch]:
    with open('data/xCodeEval/unittest_db.json', 'r') as f:
      unittests = json.load(f)
    test_batches = (unittests[uid] for uid in ids)
    return tuple(tuple((pair['input'].replace('\r\n', '\n'),
                        tuple(line.replace('\r\n', '\n') for line in pair['output']))
                 for pair in batch)
                 for batch in test_batches)


@dataclass
class XLCoST(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'c', 'cs', 'cpp', 'java', 'js', 'php', 'python',
  ]))
  _lang_to_name: dict[str, str] = field(default_factory=lambda: {
      'c': 'C',
      'cs': 'Csharp',
      'cpp': 'C++',
      'java': 'Java',
      'js': 'Javascript',
      'php': 'PHP',
      'python': 'Python',
  })

  @check_lang_support
  def _load(self, lang, column):
    lang_name = self._lang_to_name[lang]
    ds = load_dataset('codeparrot/xlcost-text-to-code', f'{lang_name}-program-level')
    return ds['train'][column]

  def load_source(self, lang: str) -> Sequence[Snippet]:
    sources = self._load(lang, 'code')
    return tuple(Snippet(str(i), code) for i, code in enumerate(sources))

  def load_tests(self, ids: Iterable[str]) -> Sequence[TestBatch]:
    raise NotImplementedError('XLCoST does not provide test cases.')


@dataclass
class CodeXGLUE(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'cs', 'java',
  ]))

  @check_lang_support
  def load_source(self, lang: str) -> Sequence[Snippet]:
    ds = load_dataset('google/code_x_glue_cc_code_to_code_trans', trust_remote_code=True)
    sources = ds['train'][lang]
    return tuple(Snippet(str(i), code) for i, code in enumerate(sources))

  def load_tests(self, ids: Iterable[str]) -> Sequence[TestBatch]:
    raise NotImplementedError('CodeXGLUE does not provide test cases.')


@dataclass
class GTransEval(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'cpp', 'java', 'python',
  ]))

  @check_lang_support
  def load_source(self, lang: str) -> Sequence[Snippet]:
    ds = load_dataset(f'xin1997/g-transeval-{lang}_all_only_input', trust_remote_code=True)
    ids = ds['train']['id']
    sources = ds['train']['content']
    return tuple(Snippet(id=id_, code=source) for id_, source in zip(ids, sources))

  def load_tests(self, ids: Iterable[str]) -> Sequence[TestBatch]:
    raise NotImplementedError('G-TransEval does not provide test cases.')


@dataclass
class CodeNet(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'java', 'cpp', 'python',
  ]))
  _lang_to_name: dict[str, str] = field(default_factory=lambda: {
      'java': 'Java',
      'cpp': 'C++',
      'python': 'Python',
  })

  @check_lang_support
  def load_source(self, lang) -> Sequence[Snippet]:
    data_dir = Path('data/Project_CodeNet/Project_CodeNet/data')
    lang_name = self._lang_to_name[lang]
    sources: list[Snippet] = []
    for subdir in data_dir.iterdir():
      if not subdir.is_dir():
        continue
      lang_dir = subdir / lang_name
      sources.extend((Snippet(id=f'{subdir.name}_{file.stem}', code=file.read_text())
                      for file in lang_dir.iterdir() if file.is_file()))
    return sources

  def load_tests(self, ids: Iterable[str]) -> Sequence[TestBatch]:
    with jsonlines.open('data/Project_CodeNet/Project_CodeNet/tests.jsonl', 'r') as reader:
      tests_dict = {obj['id']: obj['test'] for obj in reader}
    return tuple(tuple((pair[0], (pair[1],))
                       for pair in tests_dict[id_])
                 for id_ in ids)


def benchmark_factory(dataset: str) -> BaseBenchmark:
  name_to_class = {
      'HumanEvalX': HumanEvalX,
      'xCodeEval': XCodeEval,
      'XLCoST': XLCoST,
      'CodeXGLUE': CodeXGLUE,
      'G-TransEval': GTransEval,
      'CodeNet': CodeNet,
  }
  if dataset not in name_to_class:
    raise ValueError(f'{dataset} is not a valid dataset. Supported datasets: {list(name_to_class.keys())}')
  return name_to_class[dataset]()
