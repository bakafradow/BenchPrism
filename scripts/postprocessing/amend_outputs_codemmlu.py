import re
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
from tqdm import tqdm

from scripts.postprocessing import utils
from benchprism.inference import agent_factory

PATTERNS = [
    re.compile(r'\$\\boxed\{([A-D])\}\$$'),
    re.compile(r'\n+([A-D])$'),
    re.compile(r'^([A-D]).[\w ]*$'),
    re.compile(r'^([A-D])\s*\n+'),
    re.compile(r'^([A-D])\s*\([^)]+\)\s*\n+'),
    re.compile(r'^([A-D])\.(?: [\w ]*?\.?)?\s*\n+'),
    re.compile(r'the (?:most|correct) .*?(?:is|be):\s+(?:\*\*)?([A-D])'),
    re.compile(r'correct answer is ([A-D])'),
    re.compile(r'I would respond with "([A-D])"'),
]

SYS_PROMPT = """
An LLM was instructed to answer a choice question. It should have responded with a single letter, A, B, C or D. However, it failed to do so.
Given its output, your task is to recognize and extract its answer. Do NOT return anything else. If you think that part doesn't exist in the output, ONLY return `N`.

Examples:
1. Given output "The most probable behavior is a Compile Error.\n\nHere's why: ...\n\nTherefore, ...\n\nThe final answer is $\\boxed{A}$", you should return "A".
2. Given output "The most probable behavior is a Runtime Error. And", you should return "N".
"""


def ok(output: str) -> bool:
  return output in ['A', 'B', 'C', 'D']


def _amend_offline(output: str) -> str:
  global count
  for pattern in PATTERNS:
    if matched := pattern.search(output):
      count += 1
      return matched.group(1)
  return output


def amend(output: str) -> str:
  global count
  if ok(output):
    return output
  if not agent:
    return _amend_offline(output)
  res = agent.generate(sys_prompt=SYS_PROMPT, user_prompt=output)
  if res and ok(res.strip()):
    count += 1
    return res.strip()
  print(f'Failed to amend output:\n{output}')
  return output


def main():
  with jsonlines.open(args.file, 'r') as reader:
    data = list(reader)

  for row in tqdm(data, desc='Amending', total=len(data), leave=False):
    if output := row.get('output'):
      row['output'] = amend(output)
    if row.get('variant_outputs'):
      for i, output in tqdm(enumerate(row['variant_outputs']),
                            total=len(row['variant_outputs']), leave=False):
        if output:
          row['variant_outputs'][i] = amend(output)

  utils.save_with_backups(data, args.file)
  print(f'Amended {count} outputs, saved to {args.file}.')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', '--file', type=Path, required=True,
                      help='Specify the path to the jsonl file that contains model outputs.')
  parser.add_argument('-m', '--model', type=str, required=False,
                      help='Specify the model used for inference.')
  args = parser.parse_args()

  agent = agent_factory(args.model) if args.model else None
  count = 0
  main()
