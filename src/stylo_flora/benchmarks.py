import json
import re
from abc import ABC
from collections.abc import Callable, Iterable, Mapping
from collections.abc import Sequence as Seq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonlines
from datasets import load_dataset

from . import APITestCase, IOTestCase, Snippet


def check_lang_support(func: Callable) -> Callable:
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

  def load_for_translation(self, src_lang: str, dst_lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for evaluation.

    :param src_lang: the source language of the code snippets to translate
    :param dst_lang: the target language of the code snippets to translate
    :return: a sequence of source code snippets
    """
    raise NotImplementedError('Translation unsupported for current benchmark.')

  def load_for_repair(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for automatic program repair.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets
    """
    raise NotImplementedError('Repair unsupported for current benchmark.')

  def load_for_tagging(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for tag classification.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with tags
    """
    raise NotImplementedError('Tag classification unsupported for current benchmark.')

  def load_for_summarization(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for code summarization.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with human summarization
    """
    raise NotImplementedError('Code summarization unsupported for current benchmark.')

  def load_for_test_generation(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for test generation.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with test cases
    """
    raise NotImplementedError('Test generation unsupported for current benchmark.')

  def load_for_io_reasoning(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for input reasoning.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with masked assertion statements.
    """
    raise NotImplementedError('Input reasoning unsupported for current benchmark.')

  def load_for_mcq_answering(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for Multiple-Choice Question (MCQ) answering.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with multiple-choice questions and corresponding answers.
    """
    raise NotImplementedError('MCQ answering unsupported for current benchmark.')


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
  def _load(self, lang: str, task: str, column: str) -> Seq[str]:
    lang_name = self._lang_to_name[lang]
    ds = load_dataset('json', data_dir=f'data/xCodeEval/{task}/test')  # there's an issue in loading from HF when the version of datasets != 2.16.1
    ds = ds.filter(lambda row: row['lang_cluster'] == lang_name)
    return ds['train'][column]

  def _load_tests(self, ids: Iterable[str]) -> Seq[Seq[IOTestCase]]:
    with open('data/xCodeEval/unittest_db.json', 'r') as f:
      unittests = json.load(f)
    return [[IOTestCase(input=pair['input'].replace('\r\n', '\n'),
                        outputs=[output.replace('\r\n', '\n') for output in pair['output']])
             for pair in batch]
            for batch in (unittests[uid] for uid in ids)]

  def load_for_translation(self, src_lang: str, dst_lang: str) -> Seq[Snippet]:
    TASK_NAME = 'code_translation'
    src_uids = self._load(src_lang, TASK_NAME, 'src_uid')
    sources = self._load(src_lang, TASK_NAME, 'source_code')
    testcases = self._load_tests(src_uids)
    return [Snippet(id=src_uid, data={
        'code': source,
        'io_testcases': testcases[i],
    }) for i, (src_uid, source) in enumerate(zip(src_uids, sources))]

  def load_for_repair(self, lang):
    TASK_NAME = 'apr'
    src_uids = self._load(lang, TASK_NAME, 'src_uid')
    sources = self._load(lang, TASK_NAME, 'bug_source_code')
    with jsonlines.open('data/xCodeEval/problem_descriptions.jsonl', 'r') as reader:
      args_dict = {obj['src_uid']: {
        'desc': obj['description'],
        'input_spec': obj['input_spec'],
        'output_spec': obj['output_spec'],
        'sample_inputs': obj['sample_inputs'],
        'sample_outputs': obj['sample_outputs'],
      } for obj in reader}
    testcases = self._load_tests(src_uids)
    return [Snippet(id=src_uid, data={
        **args_dict[src_uid],
        'code': source,
        'io_testcases': testcases[i],
    }) for i, (src_uid, source) in enumerate(zip(src_uids, sources))]

  def load_for_tagging(self, lang: str) -> Seq[Snippet]:
    TASK_NAME = 'tag_classification'
    src_uids = self._load(lang, TASK_NAME, 'src_uid')
    sources = self._load(lang, TASK_NAME, 'source_code')
    tags_list = self._load(lang, TASK_NAME, 'tags')
    with jsonlines.open('data/xCodeEval/problem_descriptions.jsonl', 'r') as reader:
      args_dict = {obj['src_uid']: {
        'desc': obj['description'],
      } for obj in reader}
    return [Snippet(id=src_uid, data={
        'code': source,
        'tags': tags,
        'desc': args_dict[src_uid]['desc']
    }) for src_uid, source, tags in zip(src_uids, sources, tags_list)]


@dataclass
class CodeScope(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'c', 'cpp', 'cs', 'delphi', 'go', 'java', 'js', 'kotlin', 'php', 'perl', 'python', 'ruby', 'rust',
  ]))
  _lang_to_name: dict[str, str] = field(default_factory=lambda: {
      'c': 'C',
      'cpp': 'C++',
      'cs': 'C#',
      'delphi': 'Delphi',
      'go': 'Go',
      'java': 'Java',
      'js': 'JavaScript',
      'kotlin': 'Kotlin',
      'php': 'PHP',
      'perl': 'Perl',
      'python': 'Python',
      'ruby': 'Ruby',
      'rust': 'Rust',
  })

  @classmethod
  def _normalize_test(cls, testcases: str) -> Seq[IOTestCase]:
    if any(not isinstance(testcase['input'], str) and len(testcase['input']) != 1 for testcase in eval(testcases)):
      raise ValueError('Input of testcases must be a string or a sequence with length 1.')
    return [IOTestCase(input=testcase['input'].replace('\r\n', '\n') if isinstance(testcase['input'], str) \
                       else testcase['input'][0].replace('\r\n', '\n'),
                       outputs=[output.replace('\r\n', '\n') for output in testcase['output']])
            for testcase in eval(testcases)]

  @check_lang_support
  def load_for_translation(self, src_lang: str, dst_lang: str) -> Seq[Snippet]:
    ds = load_dataset('json', data_files='data/CodeScope/data/code_translation_data.jsonl')
    ds = ds.filter(lambda row: row['source_lang_cluster'] == self._lang_to_name[src_lang] and row['target_lang_cluster'] == self._lang_to_name[dst_lang])
    return [Snippet(id=row['src_uid'], data={
        'code': row['source_code'], 'io_testcases': self._normalize_test(row['testcases']),
    }) for row in ds['train']]

  @check_lang_support
  def load_for_repair(self, lang: str) -> Seq[Snippet]:
    ds = load_dataset('json', data_files='data/CodeScope/data/code_repair_data.jsonl')
    ds = ds.filter(lambda row: row['lang_cluster'] == self._lang_to_name[lang])
    return [Snippet(id=row['src_uid'], data={
        'code': row['source_code'],
        'desc': row['description'],
        'input_spec': row['input_specification'],
        'output_spec': row['output_specification'],
        'sample_inputs': row['sample_inputs'],
        'sample_outputs': row['sample_outputs'],
        'io_testcases': self._normalize_test(row['testcases']),
    }) for row in ds['train']]

  @check_lang_support
  def load_for_summarization(self, lang: str) -> Seq[Snippet]:
    ds = load_dataset('json', data_files='data/CodeScope/data/code_summarization_data.jsonl')
    ds = ds.filter(lambda row: row['lang_cluster'] == self._lang_to_name[lang])
    return [Snippet(id=row['id'], data={
        'code': row['source_code'],
        'human_summarization': row['human_summarization'],
    }) for row in ds['train']]

  @check_lang_support
  def load_for_test_generation(self, lang: str) -> Seq[Snippet]:
    ds = load_dataset('json', data_files='data/CodeScope/data/automated_testing_data.jsonl')
    ds = ds.filter(lambda row: row['lang_cluster'] == self._lang_to_name[lang])
    return [Snippet(id=row['id'], data={
        'code': row['source_code'],
        'desc': row['description'],
        'input_spec': row['input_specification'],
        'output_spec': row['output_specification'],
        'sample_inputs': row['sample_inputs'],
        'sample_outputs': row['sample_outputs'],
        'notes': row['notes'],
        'io_testcases': self._normalize_test(row['human_testcases']),  # for ensuring correct transformation
    }) for row in ds['train']]


@dataclass
class CodeMMLU(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'java', 'python',
  ]))
  _lang_to_name: dict[str, str] = field(default_factory=lambda: {
      'java': 'java',
      'python': 'python',
  })

  @check_lang_support
  def load_for_mcq_answering(self, lang):
    ds = load_dataset('Fsoft-AIC/CodeMMLU', 'execution_prediction', trust_remote_code=True)
    match lang:
      case 'java':
        ds = ds.filter(lambda row: 'public class' in row['question'])
      case 'python':
        ds = ds.filter(lambda row: 'public class' not in row['question'])
      case _:
        raise TypeError(f'Unsupported language: {lang}')
    return [Snippet(id=row['task_id'], data={
        'code': row['question'],
        'choices': row['choices'],
        'answer': row['answer'],
    }) for row in ds['test']]


@dataclass
class CoderUJB(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'java',
  ]))

  def _construct_code(self, row: Mapping[str, Any]) -> str:
    return f'{row["import_context"]}\n\n{row["class_signature"]} {{\n{row["class_field_context"]}\n\n{row["class_function_signature_context"]}\n\n{row["buggy"]}\n}}'

  @check_lang_support
  def load_for_repair(self, lang):
    ds = load_dataset('ZHENGRAN/code_ujb_repair', trust_remote_code=True)
    return [Snippet(id=row['task_id'], data={
        'code': self._construct_code(row),
        f'api_testcases_{lang}': [APITestCase(file=source['file'], code=self._construct_code(source),
                                             method=source['method'])
                                 for source in row['test_sources']],
        'oracle': row['source'],
        'start': row['start'],
        'end': row['end'],
    }) for row in ds['train']]


@dataclass
class CruxEvalX(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'java',
  ]))
  _lang_to_name: dict[str, str] = field(default_factory=lambda: {
      'java': 'Java',
  })

  @check_lang_support
  def load_for_io_reasoning(self, lang: str) -> Seq[Snippet]:
    ds = load_dataset('xhwl/cruxeval-x', trust_remote_code=True)
    return [Snippet(id=row['id'], data={
        'code': row['code'],
        'input_reasoning': row['input_reasoning'],
        'output_reasoning': row['output_reasoning'],
        'io_testcases': [IOTestCase(input='', outputs=[''])],  # tests by assertion
    }) for row in ds[self._lang_to_name[lang]]]


@dataclass
class ClassEvalT(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'cpp', 'java', 'python',
  ]))
  _lang_to_name: dict[str, str] = field(default_factory=lambda: {
      'cpp': 'cpp',
      'java': 'java',
      'python': 'py',
  })

  @check_lang_support
  def load_for_translation(self, src_lang: str, dst_lang: str) -> Seq[Snippet]:
    data_dir = Path('data/ClassEval-T/ClassEval_T')

    def get_testcase(name: str, lang: str) -> APITestCase:
      match src_lang:
        case 'cpp':
          name = name.replace('test_', '')
        case 'java':
          name = name.replace('Test', '')
        case 'python':
          ...
        case _:
          raise TypeError(f'Unsupported target language: {src_lang}')
      match lang:
        case 'cpp':
          filename = f'test_{name}.cpp'
        case 'java':
          filename = f'{name}Test.java'
        case 'python':
          filename = f'{name}.py'
        case _:
          raise TypeError(f'Unsupported language: {lang}')
      test_code_path = data_dir / self._lang_to_name[lang] / 'test' / filename
      if not test_code_path.exists():
        raise FileNotFoundError(f'Test file {test_code_path} does not exist.')
      return APITestCase(file=filename, code=test_code_path.read_text())

    src_dir = data_dir / self._lang_to_name[src_lang] / 'solution'
    if not src_dir.exists():
      raise FileNotFoundError(f'Directory {src_dir} does not exist.')
    snippets = [Snippet(id=file.stem, data={
        'code': file.read_text(),
        f'api_testcases_{src_lang}': get_testcase(file.stem, src_lang),
        f'api_testcases_{dst_lang}': get_testcase(file.stem, dst_lang),
    }) for file in src_dir.iterdir() if file.is_file() and file.suffix == f'.{self._lang_to_name[src_lang]}']
    return snippets


def benchmark_factory(dataset: str) -> BaseBenchmark:
  name_to_class = {
      'xcodeeval': XCodeEval,
      'codescope': CodeScope,
      'cruxeval-x': CruxEvalX,
      'coderujb': CoderUJB,
      'codemmlu': CodeMMLU,
      'classeval-t': ClassEvalT,
  }
  dataset = dataset.lower()
  if dataset not in name_to_class:
    raise ValueError(f'{dataset} is not a valid dataset. Supported datasets: {list(name_to_class.keys())}')
  return name_to_class[dataset]()
