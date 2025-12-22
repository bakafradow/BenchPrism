import json
import re
from abc import ABC
from collections.abc import Callable, Iterable
from collections.abc import Sequence as Seq
from hashlib import sha1
from pathlib import Path

import jsonlines
from datasets import load_dataset

from . import IOTestCase, Snippet, setting_dict
from .metrics import pass_at_1, pass_at_1_classeval


def check_lang_support(func: Callable) -> Callable:
  def wrapper(self, lang, *args, **kwargs):
    if lang not in self.supported_langs:
      raise TypeError(f'{lang} is not supported in current benchmark. Supported languages: {self.supported_langs}')
    return func(self, lang, *args, **kwargs)
  return wrapper


def _checker_io(snippet: Snippet, lang: str) -> bool:
  result = pass_at_1([snippet.data['code']], [snippet.data['io_tests']], lang=lang)
  return bool(result.pass_rate)


def _checker_classeval(snippet: Snippet, lang: str) -> bool:
  result = pass_at_1_classeval([snippet.data['code']], [snippet.data[f'test_{lang}']], lang)
  return bool(result.pass_rate_class)


class BaseBenchmark(ABC):
  """
  Abstract base class for benchmarks.
  """

  supported_langs: frozenset[str] = frozenset()
  """Supported languages in the dataset to evaluate."""
  lang_to_name: dict[str, str] = {}
  """Mapping language name from unified one to the one in dataset."""

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

  def load_for_test_generation(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for test generation.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with test cases
    """
    raise NotImplementedError('Test generation unsupported for current benchmark.')

  def load_for_summarization(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for code summarization.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with human summarization
    """
    raise NotImplementedError('Code summarization unsupported for current benchmark.')

  def load_for_mcq_answering(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for Multiple-Choice Question (MCQ) answering.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with multiple-choice questions and corresponding answers
    """
    raise NotImplementedError('MCQ answering unsupported for current benchmark.')

  def load_for_io_reasoning(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for input reasoning.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with masked assertion statements
    """
    raise NotImplementedError('IO reasoning unsupported for current benchmark.')

  def load_for_defect_detection(self, lang: str) -> Seq[Snippet]:
    """
    Loads the source code snippets for defect detection.

    :param lang: the language of the code snippets
    :return: a sequence of source code snippets with potential defects and the ground truth
    """
    raise NotImplementedError('Defect detection unsupported for current benchmark.')


class XCodeEval(BaseBenchmark):
  supported_langs = frozenset({
      'c', 'cpp', 'cs', 'go', 'java', 'js', 'kotlin', 'php', 'python', 'ruby', 'rust',
  })
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
  _data_dir = Path(setting_dict['datasets']['xcodeeval_root'])

  def __init__(self):
    super().__init__()
    if not self._data_dir.exists():
      raise FileNotFoundError(f'Please clone xCodeEval repo manually from https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval and place it to {self._data_dir}')

  def load_for_translation(self, src_lang: str, dst_lang: str) -> Seq[Snippet]:
    TASK_NAME = 'code_translation'
    code_uids = self._load(src_lang, TASK_NAME, 'code_uid')
    sources = self._load(src_lang, TASK_NAME, 'source_code')
    src_uids = self._load(src_lang, TASK_NAME, 'src_uid')
    testcases = self._load_tests(src_uids)
    return [Snippet(id=code_uid, data={
        'code': source,
        'io_tests': testcase,
        'checker': _checker_io,
    }) for code_uid, source, testcase in zip(code_uids, sources, testcases)]

  def load_for_repair(self, lang):
    TASK_NAME = 'apr'
    code_uids = self._load(lang, TASK_NAME, 'bug_code_uid')
    src_uids = self._load(lang, TASK_NAME, 'src_uid')
    sources = self._load(lang, TASK_NAME, 'bug_source_code')
    with jsonlines.open(self._data_dir / 'problem_descriptions.jsonl', 'r') as reader:
      problem_dict = {obj['src_uid']: {
          'desc': obj['description'],
          'input_spec': obj['input_spec'],
          'output_spec': obj['output_spec'],
          'sample_inputs': obj['sample_inputs'],
          'sample_outputs': obj['sample_outputs'],
      } for obj in reader}
    testcases = self._load_tests(src_uids)
    return [Snippet(id=code_uid, data={
        'code': source,
        'io_tests': testcase,
        **problem_dict[src_uid],
    }) for code_uid, src_uid, source, testcase in zip(code_uids, src_uids, sources, testcases)]

  def load_for_tagging(self, lang: str) -> Seq[Snippet]:
    TASK_NAME = 'tag_classification'
    code_uids = self._load(lang, TASK_NAME, 'code_uid')
    src_uids = self._load(lang, TASK_NAME, 'src_uid')
    sources = self._load(lang, TASK_NAME, 'source_code')
    tags_list = self._load(lang, TASK_NAME, 'tags')
    with jsonlines.open(self._data_dir / 'problem_descriptions.jsonl', 'r') as reader:
      problem_dict = {obj['src_uid']: {
          'desc': obj['description'],
      } for obj in reader}
    return [Snippet(id=code_uid, data={
        'code': source,
        'tags': tags,
        'desc': problem_dict[src_uid]['desc']
    }) for code_uid, src_uid, source, tags in zip(code_uids, src_uids, sources, tags_list)]

  @check_lang_support
  def _load(self, lang: str, task: str, column: str) -> Seq[str]:
    """
    :Note: there's an issue in loading from HF when the version of datasets != 2.16.1
    """
    lang_name = self.lang_to_name[lang]
    ds = load_dataset('json', data_dir=self._data_dir / f'{task}/test')
    ds = ds.filter(lambda row: row['lang_cluster'] == lang_name)
    return ds['train'][column]

  def _load_tests(self, ids: Iterable[str]) -> Seq[Seq[IOTestCase]]:
    with open(self._data_dir / 'unittest_db.json', 'r') as f:
      unittests = json.load(f)
    return [[IOTestCase(input=pair['input'].replace('\r\n', '\n'),
                        outputs=[output.replace('\r\n', '\n') for output in pair['output']])
             for pair in batch]
            for batch in (unittests[uid] for uid in ids)]


class CodeScope(BaseBenchmark):
  supported_langs = frozenset({
      'c', 'cpp', 'cs', 'delphi', 'go', 'java', 'js', 'kotlin', 'php', 'perl', 'python', 'ruby', 'rust',
  })
  lang_to_name = {
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
  }
  _data_dir = Path(setting_dict['datasets']['codescope_root'])

  def __init__(self):
    super().__init__()
    if not self._data_dir.exists():
      raise FileNotFoundError(f'Please clone CodeScope repo manually from https://github.com/WeixiangYAN/CodeScope and place it to {self._data_dir}')

  @check_lang_support
  def load_for_translation(self, src_lang: str, dst_lang: str) -> Seq[Snippet]:
    ds = load_dataset('json', data_files=str(self._data_dir / 'data/code_translation_data.jsonl'))
    ds = ds.filter(lambda row: row['source_lang_cluster'] == self.lang_to_name[src_lang] and row['target_lang_cluster'] == self.lang_to_name[dst_lang])
    return [Snippet(id=row['src_uid'], data={
        'code': row['source_code'],
        'io_tests': self._normalize_test(row['testcases']),
        'checker': _checker_io,
    }) for row in ds['train']]

  @check_lang_support
  def load_for_repair(self, lang: str) -> Seq[Snippet]:
    ds = load_dataset('json', data_files=str(self._data_dir / 'data/code_repair_data.jsonl'))
    ds = ds.filter(lambda row: row['lang_cluster'] == self.lang_to_name[lang])
    return [Snippet(id=row['src_uid'], data={
        'code': row['source_code'],
        'desc': row['description'],
        'input_spec': row['input_specification'],
        'output_spec': row['output_specification'],
        'sample_inputs': row['sample_inputs'],
        'sample_outputs': row['sample_outputs'],
        'io_tests': self._normalize_test(row['testcases']),
    }) for row in ds['train']]

  @check_lang_support
  def load_for_test_generation(self, lang: str) -> Seq[Snippet]:
    ds = load_dataset('json', data_files=str(self._data_dir / 'data/automated_testing_data.jsonl'))
    ds = ds.filter(lambda row: row['lang_cluster'] == self.lang_to_name[lang])
    return [Snippet(id=row['id'], data={
        'code': row['source_code'],
        'desc': row['description'],
        'input_spec': row['input_specification'],
        'output_spec': row['output_specification'],
        'sample_inputs': row['sample_inputs'],
        'sample_outputs': row['sample_outputs'],
        'notes': row['notes'],
        'io_tests': self._normalize_test(row['human_testcases']),  # used by the checker
        'checker': _checker_io,
    }) for row in ds['train']]

  @check_lang_support
  def load_for_summarization(self, lang: str) -> Seq[Snippet]:
    ds = load_dataset('json', data_files=str(self._data_dir / 'data/code_summarization_data.jsonl'))
    ds = ds.filter(lambda row: row['lang_cluster'] == self.lang_to_name[lang])
    return [Snippet(id=row['id'], data={
        'code': row['source_code'],
        'human_summarization': row['human_summarization'],
    }) for row in ds['train']]

  @staticmethod
  def _normalize_test(testcases: str) -> Seq[IOTestCase]:
    if any(not isinstance(testcase['input'], str) and len(testcase['input']) != 1 for testcase in eval(testcases)):
      raise ValueError('Input of testcases must be a string or a sequence with length 1.')
    return [IOTestCase(input=testcase['input'].replace('\r\n', '\n') if isinstance(testcase['input'], str)
                       else testcase['input'][0].replace('\r\n', '\n'),
                       outputs=[output.replace('\r\n', '\n') for output in testcase['output']])
            for testcase in eval(testcases)]


class CodeMMLU(BaseBenchmark):
  supported_langs = frozenset({
      'java', 'python',
  })
  lang_to_name = {
      'java': 'java',
      'python': 'python',
  }

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


class CruxEvalX(BaseBenchmark):
  supported_langs = frozenset({
      'java',
  })
  _lang_to_name = {
      'java': 'Java',
  }

  @check_lang_support
  def load_for_io_reasoning(self, lang: str) -> Seq[Snippet]:
    ds = load_dataset('xhwl/cruxeval-x', trust_remote_code=True)
    return [Snippet(id=row['id'], data={
        'code': row['code'],
        'input_reasoning': row['input_reasoning'],
        'output_reasoning': row['output_reasoning'],
        'io_tests': [IOTestCase(input='', outputs=[''])],  # tests by assertion
        'checker': _checker_io,
    }) for row in ds[self._lang_to_name[lang]]]


class ClassEvalT(BaseBenchmark):
  """
  :Note: Problematic snippets that cannot pass unit tests originally:
  - `SQLQueryBuilder` (with index `69`) in Java;
  - `CookiesUtil`, `IpUtil`, `SignInSystem`, and `VendingMachine` (with indices `26, 45, 72, 88`) in C++;
  - `BookManagementDB`, `IpUtil`, `KappaCalculator`, `MovieTicketDB`, `StudentDatabaseProcessor`, `UserLoginDB` (with indices `14, 45, 48, 55, 77, 86`) in Python.
  """

  supported_langs = frozenset({
      'cpp', 'java', 'python',
  })
  lang_to_name = {
      'cpp': 'cpp',
      'java': 'java',
      'python': 'py',
  }
  _data_dir = Path(setting_dict['datasets']['classeval_t_root'])

  def __init__(self):
    super().__init__()
    if not self._data_dir.exists():
      raise FileNotFoundError(f'Please clone ClassEval-T repo manually from https://github.com/wLinHoo/ClassEval-T and place it to {self._data_dir} (and run the amending script).')

  @check_lang_support
  def load_for_translation(self, src_lang: str, dst_lang: str) -> Seq[Snippet]:
    src_dir = self._data_dir / 'ClassEval_T' / self.lang_to_name[src_lang] / 'solution'
    if not src_dir.exists():
      raise FileNotFoundError(f'Directory {src_dir} does not exist.')
    return [Snippet(id=file.stem, data={
        'code': file.read_text(),
        f'test_{src_lang}': self._load_test(file.stem, src_lang),
        f'test_{dst_lang}': self._load_test(file.stem, dst_lang),
        'checker': _checker_classeval,
    }) for file in sorted(src_dir.iterdir())
        if file.is_file() and file.suffix == f'.{self.lang_to_name[src_lang]}']

  def _load_test(self, name: str, lang: str) -> str:
    soln_name = self._normalize_name(name, lang)
    match lang:
      case 'cpp':
        test_name = f'test_{soln_name}.cpp'
      case 'java':
        test_name = f'{soln_name}Test.java'
      case 'python':
        test_name = f'{soln_name}.py'
      case _:
        raise TypeError(f'Unsupported language: {lang}')
    test_code_path = self._data_dir / 'ClassEval_T' / self.lang_to_name[lang] / 'test' / test_name
    if not test_code_path.exists():
      raise FileNotFoundError(f'Test file {test_code_path} does not exist.')
    return test_code_path.read_text()

  @staticmethod
  def _normalize_name(name: str, lang: str) -> str:
    match lang:
      case 'cpp':
        return name.replace('test_', '')
      case 'java':
        return name.replace('Test', '')
      case 'python':
        return name
      case _:
        raise TypeError(f'Unsupported language: {lang}')


class CoderUJB(BaseBenchmark):
  supported_langs = frozenset({
      'java',
  })
  _data_dir = Path(setting_dict['datasets']['coderujb_root'])

  @staticmethod
  def _extract_prefix(prompt: str) -> str:
    matched = re.search(r'(^.+```java\n\s*(?:(?:\/\/[^\n]*|\/\*.*?\*\/)\s*)*\n)(.+)\n```\s*(?:\Z|Please)', prompt, re.S)
    if not matched:
      raise ValueError(f'Failed to extract prefix from prompt: {prompt}')
    return matched.group(1)

  @check_lang_support
  def load_for_repair(self, lang: str) -> Seq[Snippet]:
    ds = load_dataset('ZHENGRAN/code_ujb_repair', trust_remote_code=True)
    return [Snippet(id=row['task_id'], data={
        'code': f'{row["class_signature"]}{{\n{row["buggy"]}\n}}',
        'prompt_prefix': self._extract_prefix(row['prompt_chat']),
        'function_signature': row['function_signature'],
        'project': row['project'],
        'bug_id': row['bug_id'],
        'source': row['source'],
        'start': row['start'],
        'end': row['end'],
        'location': row['location'],
        'testmethods': row['testmethods'],
    }) for row in ds['train']]

  @check_lang_support
  def load_for_defect_detection(self, lang: str) -> Seq[Snippet]:
    """
    :Note: In the HF dataset, `prompt_chat` field stands for FS0 prompt without program context. Thus we use manually constructed dataset instead.
    """
    with open(self._data_dir / 'datasets' / 'data' /
              'task_defectdetection_bench_1111|2048.json', 'r') as f:
      ds = json.load(f)
    return [Snippet(id=f'{row["task_id"]}-{row["bug_id"]}', data={
        'code': f'public class Dummy {{\n{row["code"]}\n}}',
        'prompt_prefix': self._extract_prefix(row['prompt_chat']),
        'function_signature': row['function_signature'],
        'defective': row['defective'],
    }) for row in ds['code_ujb_defectdetection']]


class TestBench(BaseBenchmark):
  supported_langs = frozenset({
      'java',
  })
  _data_dir = Path(setting_dict['datasets']['testbench_root'])

  def __init__(self):
    super().__init__()
    if not self._data_dir.exists():
      raise FileNotFoundError(f'Please clone TestBench repo manually from https://github.com/iSEngLab/TestBench and place it to {self._data_dir}.')

  @check_lang_support
  def load_for_test_generation(self, lang: str) -> Seq[Snippet]:
    snippets: list[Snippet] = []
    for file in sorted((self._data_dir / 'source_file_parser').iterdir()):
      if file.suffix != '.json':
        continue
      with open(file, 'r') as f:
        data = json.load(f)
      snippets.extend([Snippet(id=self._make_id(row), data={
          'code': f'public class Dummy {{\n{row["source_code"]}\n}}',
          'full_context': row['full_context'],
          'simple_context': row['simple_context'],
          'project_name': row['project_name'],
          'package': row['package'],
          'class_name': row['class_name'],
          'method_name': row['method_name'],
          'relative_path': row['relative_path'],
      }) for row in data])
    return snippets

  @staticmethod
  def _make_id(row: dict[str, str]) -> str:
    id_str = f'{row["project_name"]}:{row["class_name"]}:{row["source_code"]}'
    return sha1(id_str.encode('utf-8')).hexdigest()


def benchmark_factory(dataset: str) -> BaseBenchmark:
  name_to_class = {
      'xcodeeval': XCodeEval,
      'codescope': CodeScope,
      'cruxeval-x': CruxEvalX,
      'codemmlu': CodeMMLU,
      'classeval-t': ClassEvalT,
      'coderujb': CoderUJB,
      'testbench': TestBench,
  }
  dataset = dataset.lower()
  if dataset not in name_to_class:
    raise ValueError(f'{dataset} is not a valid dataset. Supported datasets: {list(name_to_class.keys())}')
  return name_to_class[dataset]()
