"""
Modified from CoderUJB repository (https://github.com/ZZR0/CoderUJB).
"""

from argparse import ArgumentParser
from pathlib import Path

import jsonlines

from scripts.experiment.run import SUPPORTED_TASKS
from scripts.experiment.utils import load_snippets
from scripts.postprocessing import utils


def _remove_line_comment(signature):
  pure_signature = ''
  line_comment = False
  for idx, c in enumerate(signature):
    if c == '/' and idx < len(signature) - 1 and signature[idx + 1] == '/':
      line_comment = True
    if line_comment:
      if c == '\n':
        line_comment = False
      continue
    pure_signature += c
  return pure_signature


def _clean_signature(signature):
  signature = _remove_line_comment(signature)
  if signature.startswith('@'):
    for idx, c in enumerate(signature):
      if c == ' ' or c == '\n':
        break
    pre_signature = signature[:idx] + '\n'
    sub_signature = signature[idx:].strip()
    sub_signature = sub_signature.split('(')[0].strip()
  else:
    pre_signature = ''
    sub_signature = signature.split('(')[0]
  return pre_signature, sub_signature


def _stop_at_function(generation):
  block_count, in_block, in_double_quote, in_single_quote = 0, False, False, False
  for i, ch in enumerate(generation):
    if ch == '"' and (i == 0 or generation[i - 1] != '\\'):
      in_double_quote = not in_double_quote
    if ch == '\'' and (i == 0 or generation[i - 1] != '\\'):
      in_single_quote = not in_single_quote
    if ch == '{' and (not in_double_quote):
      block_count += 1
      in_block = True
    if ch == '}' and (not in_double_quote):
      block_count -= 1
    if block_count == 0 and in_block:
      break
  if i:
    generation = generation[:i + 1]
  return generation


def postprocess_generation_chat(generation: str, signature: str) -> str:
  global count
  pre_signature, sub_signature = _clean_signature(signature)
  if sub_signature not in generation:
    print(f'Target function `{sub_signature}` not found in answer.')
    return generation
  parts = generation.split(sub_signature)
  function = _stop_at_function(parts[1])

  patch = pre_signature + sub_signature + function
  count += patch != generation
  return patch


def main():
  with jsonlines.open(args.file, 'r') as reader:
    data = list(reader)

  for row in data:
    signature = snippet_dict[row['id']].data['function_signature']
    if output := row.get('output'):
      row['output'] = postprocess_generation_chat(output, signature)
    if row.get('variant_outputs'):
      for i, output in enumerate(row['variant_outputs']):
        if output:
          row['variant_outputs'][i] = postprocess_generation_chat(output, signature)

  utils.save_with_backups(data, args.file)
  print(f'Amended {count} outputs, saved to {args.file}.')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', '--file', type=Path, required=True,
                      help='Specify the path to the jsonl file that contains model outputs.')
  args = parser.parse_args()
  args.dataset = 'coderujb'
  args.task = 'code_repair'
  args.src_lang = 'java'

  snippets = load_snippets(args)
  snippet_dict = {snippet.id: snippet for snippet in snippets}
  count = 0
  main()
