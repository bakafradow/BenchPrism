import re
from argparse import ArgumentParser
from pathlib import Path
from reprlib import repr

import jsonlines
from tqdm import tqdm

SYS_PROMPT = """
An LLM was instructed to translate a code snippet. Given its output, i.e. the translated code, and unit tests in the same language, your task is to align the class/method names in the translated code with the unit tests.
For example, if the unit tests call `add_course_score` while the translated code declares `addCourseScore` method, you should replace `addCourseScore` in the translated code with `add_course_score`.
You should only consider probable API inconsistency, not functionality of the code. Your output MUST only contain the modified translated code WITHOUT any explanation, enclosed by triple back quotes with the language specified.
"""


def _amend_with_lm(output: str, snippet: 'Snippet', name: str) -> str:
  global count
  user_prompt = (f'Translated code:\n```{args.dst_lang}\n{output}```\n'
                 f'Unit Tests:\n```{args.dst_lang}\n{snippet.data[f"test_{args.dst_lang}"]}```')
  res = agent.generate(sys_prompt=SYS_PROMPT, user_prompt=user_prompt)
  if res:
    code = task.resolve_response(res)
    if code:
      count += 1
      return code
    print(f'Failed to resolve response for {name}: {repr(res)}')
  else:
    print(f'Failed to generate response for {name}: {repr(output)}')
  return output


PATTERN_SNAKE_CLASS_NAME = re.compile(r'^\s*class\s+([a-z0-9]+(?:_[a-z0-9]+)*)', re.M)
PATTERN_CAMEL_FIELD_NAME_CPP = re.compile(r'^[A-Za-z0-9_:&*<>, \t]+\s+\*?([a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+)\b', re.M)
PATTERN_CAMEL_FIELD_NAME_PYTHON = re.compile(r'(?:def\s+|self\.)(_?[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+)')
PATTERN_CAMEL_INTERSPACE = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z0-9])')


def _class_name_to_camel(code: str) -> str:
  for match in set(PATTERN_SNAKE_CLASS_NAME.findall(code)):
    new_name = ''.join([word.capitalize() for word in match.split('_')])
    code = re.sub(rf'\b{match}\b', new_name, code)
  return code


def _field_name_to_snake_cpp(code: str) -> str:
  for match in set(PATTERN_CAMEL_FIELD_NAME_CPP.findall(code)):
    new_name = PATTERN_CAMEL_INTERSPACE.sub('_', match).lower()
    code = re.sub(rf'\b{match}\b', new_name, code)
  return code


def _field_name_to_snake_python(code: str) -> str:
  for match in set(PATTERN_CAMEL_FIELD_NAME_PYTHON.findall(code)):
    new_name = PATTERN_CAMEL_INTERSPACE.sub('_', match).lower()
    code = re.sub(rf'\b{match}\b', new_name, code)
  return code


def _amend_with_re(output: str) -> str:
  global count
  code = _class_name_to_camel(output)
  if args.dst_lang == 'cpp':
    code = _field_name_to_snake_cpp(code)
  elif args.dst_lang == 'python':
    code = _field_name_to_snake_python(code)
  if code != output:
    count += 1
  return code


def amend(output: str, id_: str, seq: int | None = None) -> str:
  if not args.model:
    return _amend_with_re(output)
  try:
    snippet = next(snippet for snippet in snippets if snippet.id == id_).replace(code=output)
  except StopIteration:
    print(f'No snippet with id {id_}')
    return output
  if snippet.data['checker'](snippet, args.dst_lang):
    return output
  name = f'{id_}_{(str(seq) if seq else "orig")}'
  return _amend_with_lm(output, snippet, name)


def main():
  with jsonlines.open(args.file, 'r') as reader:
    data = list(reader)

  for row in tqdm(data, desc='Amending', total=len(data), leave=False):
    id_ = row['id']
    if output := row.get('output'):
      row['output'] = amend(output, id_)
    if row.get('variant_outputs'):
      for i, output in tqdm(enumerate(row['variant_outputs']),
                            total=len(row['variant_outputs']), leave=False):
        if output:
          row['variant_outputs'][i] = amend(output, id_, i)

  amended_file = args.file.with_stem('amended_' + args.file.stem)
  with jsonlines.open(amended_file, mode='w') as writer:
    writer.write_all(data)
  print(f'Amended {count} outputs, saved to {amended_file}.')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', '--file', type=Path, required=True,
                      help='Specify the path to the jsonl file that contains model outputs.')
  parser.add_argument('-m', '--model', type=str, required=False,
                      help='Specify the model used for inference. If not set, amends by text substitution')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('--dst-lang', type=str, required=True,
                      help='Specify the destination language.')
  args = parser.parse_args()

  if args.model:
    from stylo_flora import Snippet
    from stylo_flora.benchmarks import ClassEvalT
    from stylo_flora.inference import agent_factory
    from stylo_flora.inference.tasks import CodeTranslation

    benchmark = ClassEvalT()
    snippets = benchmark.load_for_translation(args.src_lang, args.dst_lang)
    agent = agent_factory(args.model)
    task = CodeTranslation(args.src_lang, args.dst_lang)
  count = 0
  main()
