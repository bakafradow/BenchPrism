import json
from abc import ABC
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jsonlines
from datasets import load_dataset

from . import Snippet, TestBatch


def check_lang_support(func: Callable) -> Callable:
  def wrapper(self, task, lang, *args, **kwargs):
    if lang not in self.supported_langs:
      raise TypeError(f'{kwargs["lang"]} is not supported in current benchmark. Supported languages: {self.supported_langs}')
    return func(self, task, lang, *args, **kwargs)
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

  def load_for_translation(self, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
    """
    Loads the source code snippets for evaluation.

    :param src_lang: the source language of the code snippets to translate
    :param dst_lang: the target language of the code snippets to translate
    :return: a sequence of source code snippets
    """
    raise NotImplementedError('Translation unsupported for current benchmark.')

  def load_for_apr(self, lang: str) -> Sequence[Snippet]:
    """
    Loads the source code snippets for automatic program repair.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets
    """
    raise NotImplementedError('APR unsupported for current benchmark.')

  def load_for_tagging(self, lang: str) -> Sequence[Snippet]:
    """
    Loads the source code snippets for tag classification.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with tags
    """
    raise NotImplementedError('Tag classification unsupported for current benchmark.')


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
  def _load(self, task, lang, column) -> Sequence[str]:
    lang_name = self._lang_to_name[lang]
    ds = load_dataset('json', data_dir=f'data/xCodeEval/{task}/test')  # there's an issue in loading from HF when the version of datasets != 2.16.1
    ds = ds.filter(lambda row: row['lang_cluster'] == lang_name)
    return ds['train'][column]

  def _load_tests(self, ids: Iterable[str]) -> Sequence[TestBatch]:
    with open('data/xCodeEval/unittest_db.json', 'r') as f:
      unittests = json.load(f)
    return [[(pair['input'].replace('\r\n', '\n'),
              [output.replace('\r\n', '\n') for output in pair['output']])
             for pair in batch]
            for batch in (unittests[uid] for uid in ids)]

  def load_for_translation(self, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
    TASK_NAME = 'code_translation'
    src_uids = self._load(TASK_NAME, src_lang, 'src_uid')
    sources = self._load(TASK_NAME, src_lang, 'source_code')
    testcases = self._load_tests(src_uids)
    return [Snippet(id=src_uid, code=source, args={
        'testcases': testcases[i],
    }) for i, (src_uid, source) in enumerate(zip(src_uids, sources))]

  def load_for_apr(self, lang):
    TASK_NAME = 'apr'
    src_uids = self._load(TASK_NAME, lang, 'src_uid')
    sources = self._load(TASK_NAME, lang, 'bug_source_code')
    with jsonlines.open('data/xCodeEval/problem_descriptions.jsonl', 'r') as reader:
      args_dict = {obj['src_uid']: {
        'desc': obj['description'],
        'input_spec': obj['input_spec'],
        'output_spec': obj['output_spec'],
        'sample_inputs': obj['sample_inputs'],
        'sample_outputs': obj['sample_outputs'],
      } for obj in reader}
    testcases = self._load_tests(src_uids)
    return [Snippet(id=src_uid, code=source, args={**args_dict[src_uid], 'testcases': testcases[i]})
            for i, (src_uid, source) in enumerate(zip(src_uids, sources))]

  def load_for_tagging(self, lang: str) -> Sequence[Snippet]:
    TASK_NAME = 'tag_classification'
    src_uids = self._load(TASK_NAME, lang, 'src_uid')
    sources = self._load(TASK_NAME, lang, 'source_code')
    tags_list = self._load(TASK_NAME, lang, 'tags')
    with jsonlines.open('data/xCodeEval/problem_descriptions.jsonl', 'r') as reader:
      args_dict = {obj['src_uid']: {
        'desc': obj['description'],
      } for obj in reader}
    return [Snippet(id=src_uid, code=source, args={'tags': tags, 'desc': args_dict[src_uid]['desc']})
            for src_uid, source, tags in zip(src_uids, sources, tags_list)]

  def load_tests(self, ids: Iterable[str]) -> Sequence[TestBatch]:
    with open('data/xCodeEval/unittest_db.json', 'r') as f:
      unittests = json.load(f)
    test_batches = (unittests[uid] for uid in ids)
    return [[(pair['input'].replace('\r\n', '\n'),
              [line.replace('\r\n', '\n') for line in pair['output']])
            for pair in batch]
            for batch in test_batches]


@dataclass
class HumanEvalX(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'python', 'cpp', 'go', 'java', 'js',
  ]))

  @check_lang_support
  def _load(self, lang: str, column: str) -> Sequence[str]:
    ds = load_dataset('THUDM/humaneval-x', lang, trust_remote_code=True)
    return [row[column] for row in ds['test']]

  def load_for_translation(self, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
    task_ids = self._load(src_lang, 'task_id')
    declarations = self._load(src_lang, 'declaration')
    bodies = self._load(src_lang, 'canonical_solution')
    entries = self._load(src_lang, 'test')
    sources = (f'{declaration}\n{body}\n{entry}' for declaration, body, entry in zip(declarations, bodies, entries))
    return [Snippet(id=task_id, code=source, args={
        'testcases': (('', ('',)),)  # HumanEvalX tests by assertion
    }) for task_id, source in zip(task_ids, sources)]


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

  def load_for_translation(self, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
    sources = self._load(src_lang, 'code')
    return [Snippet(str(i), code) for i, code in enumerate(sources)]


@dataclass
class CodeXGLUE(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'cs', 'java',
  ]))

  @check_lang_support
  def load_for_translation(self, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
    ds = load_dataset('google/code_x_glue_cc_code_to_code_trans', trust_remote_code=True)
    sources = ds['train'][src_lang]
    return [Snippet(str(i), code) for i, code in enumerate(sources)]


@dataclass
class GTransEval(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'cpp', 'java', 'python',
  ]))

  @check_lang_support
  def load_for_translation(self, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
    ds = load_dataset(f'xin1997/g-transeval-{src_lang}_all_only_input', trust_remote_code=True)
    ids = ds['train']['id']
    sources = ds['train']['content']
    return [Snippet(id=id_, code=source) for id_, source in zip(ids, sources)]


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

  def _load_tests(self) -> Mapping[str, TestBatch]:
    with jsonlines.open('data/Project_CodeNet/Project_CodeNet/tests.jsonl', 'r') as reader:
      return {obj['id']: obj['test'] for obj in reader}

  @check_lang_support
  def load_for_translation(self, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
    data_dir = Path('data/Project_CodeNet/Project_CodeNet/data')
    lang_name = self._lang_to_name[src_lang]
    test_dict = self._load_tests()
    snippets: list[Snippet] = []
    for subdir in data_dir.iterdir():
      if not subdir.is_dir():
        continue
      lang_dir = subdir / lang_name
      snippets.extend((Snippet(id=f'{subdir.name}_{file.stem}', code=file.read_text(), args={
          'testcases': test_dict[subdir.name],
      }) for file in lang_dir.iterdir() if file.is_file()))
    return snippets


def benchmark_factory(dataset: str) -> BaseBenchmark:
  name_to_class = {
      'xCodeEval': XCodeEval,
      'CodeScope': CodeScope,
      'HumanEvalX': HumanEvalX,
      'XLCoST': XLCoST,
      'CodeXGLUE': CodeXGLUE,
      'G-TransEval': GTransEval,
      'CodeNet': CodeNet,
  }
  if dataset not in name_to_class:
    raise ValueError(f'{dataset} is not a valid dataset. Supported datasets: {list(name_to_class.keys())}')
  return name_to_class[dataset]()
