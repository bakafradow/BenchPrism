import json
import re
from abc import ABC
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jsonlines
from datasets import load_dataset

from . import Snippet, TestBatch


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

  def load_for_summarization(self, lang: str) -> Sequence[Snippet]:
    """
    Loads the source code snippets for code summarization.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with human summarization
    """
    raise NotImplementedError('Code summarization unsupported for current benchmark.')

  def load_for_io_reasoning(self, lang: str) -> Sequence[Snippet]:
    """
    Loads the source code snippets for input reasoning.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with masked assertion statements.
    """
    raise NotImplementedError('Input reasoning unsupported for current benchmark.')

  def load_for_mcq_answering(self, lang: str) -> Sequence[Snippet]:
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
  def _load(self, lang: str, task: str, column: str) -> Sequence[str]:
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
    src_uids = self._load(src_lang, TASK_NAME, 'src_uid')
    sources = self._load(src_lang, TASK_NAME, 'source_code')
    testcases = self._load_tests(src_uids)
    return [Snippet(id=src_uid, code=source, args={
        'testcases': testcases[i],
    }) for i, (src_uid, source) in enumerate(zip(src_uids, sources))]

  def load_for_apr(self, lang):
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
    return [Snippet(id=src_uid, code=source, args={**args_dict[src_uid], 'testcases': testcases[i]})
            for i, (src_uid, source) in enumerate(zip(src_uids, sources))]

  def load_for_tagging(self, lang: str) -> Sequence[Snippet]:
    TASK_NAME = 'tag_classification'
    src_uids = self._load(lang, TASK_NAME, 'src_uid')
    sources = self._load(lang, TASK_NAME, 'source_code')
    tags_list = self._load(lang, TASK_NAME, 'tags')
    with jsonlines.open('data/xCodeEval/problem_descriptions.jsonl', 'r') as reader:
      args_dict = {obj['src_uid']: {
        'desc': obj['description'],
      } for obj in reader}
    return [Snippet(id=src_uid, code=source, args={'tags': tags, 'desc': args_dict[src_uid]['desc']})
            for src_uid, source, tags in zip(src_uids, sources, tags_list)]


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
  def _normalize_test(cls, testcases: str) -> Sequence[tuple[str, Sequence[str]]]:
    if any(not isinstance(testcase['input'], str) and len(testcase['input']) != 1 for testcase in eval(testcases)):
      raise ValueError('Input of testcases must be a string or a sequence with length 1.')
    return [(testcase['input'].replace('\r\n', '\n') if isinstance(testcase['input'], str) \
             else testcase['input'][0].replace('\r\n', '\n'),
             [output.replace('\r\n', '\n') for output in testcase['output']])
            for testcase in eval(testcases)]

  @check_lang_support
  def load_for_translation(self, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
    ds = load_dataset('json', data_files='data/CodeScope/data/code_translation_data.jsonl')
    ds = ds.filter(lambda row: row['source_lang_cluster'] == self._lang_to_name[src_lang] and row['target_lang_cluster'] == self._lang_to_name[dst_lang])
    return [Snippet(id=row['src_uid'], code=row['source_code'], args={
        'testcases': self._normalize_test(row['testcases']),
    }) for row in ds['train']]

  @check_lang_support
  def load_for_apr(self, lang: str) -> Sequence[Snippet]:
    ds = load_dataset('json', data_files='data/CodeScope/data/code_repair_data.jsonl')
    ds = ds.filter(lambda row: row['lang_cluster'] == self._lang_to_name[lang])
    return [Snippet(id=row['src_uid'], code=row['source_code'], args={
        'desc': row['description'],
        'input_spec': row['input_specification'],
        'output_spec': row['output_specification'],
        'sample_inputs': row['sample_inputs'],
        'sample_outputs': row['sample_outputs'],
        'testcases': self._normalize_test(row['testcases']),
    }) for row in ds['train']]

  @check_lang_support
  def load_for_summarization(self, lang: str) -> Sequence[Snippet]:
    ds = load_dataset('json', data_files='data/CodeScope/data/code_summarization_data.jsonl')
    ds = ds.filter(lambda row: row['lang_cluster'] == self._lang_to_name[lang])
    return [Snippet(id=row['id'], code=row['source_code'], args={
        'human_summarization': row['human_summarization'],
    }) for row in ds['train']]


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
  def load_for_translation(self, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
    src_dir = Path(f'data/ClassEval-T/ClassEval_T/{self._lang_to_name[src_lang]}/solution')
    test_dir = Path(f'data/ClassEval-T/ClassEval_T/{self._lang_to_name[dst_lang]}/test')
    if not src_dir.exists() or not test_dir.exists():
      raise FileNotFoundError(f'Directory {src_dir} or {test_dir} does not exist.')

    def get_tester(name: str) -> str:
      match dst_lang:
        case 'cpp':
          filename = f'test_{name}.cpp'
        case 'java':
          filename = f'{name}Test.java'
        case 'python':
          filename = f'{name}.py'
        case _:
          raise TypeError(f'Unsupported target language: {dst_lang}')
      tester_path = test_dir / filename
      if not tester_path.exists():
        raise FileNotFoundError(f'Test file {tester_path} does not exist.')
      return tester_path.read_text()

    snippets = [Snippet(id=file.stem, code=file.read_text(), args={
        'tester': get_tester(file.stem),
    }) for file in src_dir.iterdir() if file.is_file() and file.suffix == f'.{self._lang_to_name[src_lang]}']
    return snippets


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
    return [Snippet(id=row['task_id'], code=row['question'], args={
        'choices': row['choices'],
        'answer': row['answer'],
    }) for row in ds['test']]


@dataclass
class CruxEvalX(BaseBenchmark):
  _supported_langs: frozenset[str] = field(default_factory=lambda: frozenset([
      'java',
  ]))
  _lang_to_name: dict[str, str] = field(default_factory=lambda: {
      'java': 'Java',
  })

  def _remove_main(self, lang: str, code: str) -> str:
    match lang:
      case 'java':
        return re.sub(r'\s+public\s+static\s+void\s+main.*$', '\n}', code, flags=re.DOTALL)
      case _:
        raise TypeError(f'Unsupported language: {lang}')

  @check_lang_support
  def load_for_io_reasoning(self, lang: str) -> Sequence[Snippet]:
    ds = load_dataset('xhwl/cruxeval-x', trust_remote_code=True)
    return [Snippet(id=row['id'], code=self._remove_main(lang, row['code']), args={
        'input_reasoning': row['input_reasoning'],
        'output_reasoning': row['output_reasoning'],
        'testcases': (('', ('',)),)  # tests by assertion
    }) for row in ds[self._lang_to_name[lang]]]


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
        'testcases': (('', ('',)),)  # tests by assertion
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
      'xcodeeval': XCodeEval,
      'codescope': CodeScope,
      'classeval-t': ClassEvalT,
      'cruxeval-x': CruxEvalX,
      'codemmlu': CodeMMLU,
      'humaneval-X': HumanEvalX,
      'xlcost': XLCoST,
      'codexglue': CodeXGLUE,
      'g-transeval': GTransEval,
      'codenet': CodeNet,
  }
  dataset = dataset.lower()
  if dataset not in name_to_class:
    raise ValueError(f'{dataset} is not a valid dataset. Supported datasets: {list(name_to_class.keys())}')
  return name_to_class[dataset]()
