import re
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
from tqdm import tqdm

from scripts.postprocessing import utils
from stylo_flora.inference import agent_factory

SYS_PROMPT = """
An LLM was instructed to reason about the missing part of a program. Specifically, given a Java assertion statement `assert(f(x).equals(y))` or `assert(f(x) == y)`, It should have responded with the exact content of `{part}`. However, it failed to do so.
Given its output, your task is to fix its answer. A very common fix is to strip `x` from function call `f(x)`. Do NOT return anything else. If you think that part doesn't exist in the output, ONLY return one word `N`.

Examples:
1. Given output "f(\"w\", \")\", 13)", you should return "\"w\", \")\", 13" (strip the function call).
2. Given output "\"34\"\n```\n```java\nimport java.lang.String;  // This import is not needed but should be mentioned for completeness in real usage", you should return "\"34\"" (remove unrelevant parts).
3. Given output "\"new ArrayList<>(Arrays.asList(\\\"a\\\"))\"", you should return "new ArrayList<>(Arrays.asList(\"a\"))" (avoid unnecessary stringification).
"""

PATTERN_ASSERT = re.compile(r'assert\s*\(?\s*f\s*\((.*)\)\s*(?:.\s*equals\s*\((.*)\)|==\s*(.*))\s*\)?', re.S)
PATTERN_F = re.compile(r'\bf\s*\(', re.S)
PATTERN_STR = re.compile(r'^"(.*\\".*)"$', re.S)


def ok(output: str, group_idx: int) -> bool:
  return not (any(word in output for word in ['```', 'equals', 'java', '\\\\\\\"']) or
              PATTERN_F.search(output) or
              PATTERN_STR.match(output))


def _amend_offline(output: str, group_idx: int) -> str:
  global count
  if matched := PATTERN_ASSERT.search(output):
    count += 1
    return matched.group(group_idx)
  return output


def _amend_interactive(output: str, group_idx: int) -> str:
  global count
  if (matched := PATTERN_STR.match(output)) and input(f' {output} [y/N]: ') == 'y':
      count += 1
      return matched.group(1).encode('utf-8').decode('unicode_escape')
  return output


def amend(output: str, group_idx: int) -> str:
  global count
  if ok(output, group_idx):
    return output
  if args.interactive:
    return _amend_interactive(output, group_idx)
  if not agent:
    return _amend_offline(output, group_idx)
  part = 'x' if group_idx == 1 else 'y'
  res = agent.generate(sys_prompt=SYS_PROMPT.format(part=part), user_prompt=output)
  if not res:
    print(f'Failed to generate response:\n{output}')
    return output
  res = res.strip()
  if ok(res, group_idx):
    count += 1
    return res
  print(f'Failed to resolve response. Output:\n{output}\nResponse:\n{res}')
  return output


def main():
  with jsonlines.open(args.file, 'r') as reader:
    data = list(reader)

  for row in tqdm(data, desc='Amending', total=len(data), leave=False):
    if output := row.get('output'):
      row['output'] = amend(output, group_idx)
    if row.get('variant_outputs'):
      for i, output in tqdm(enumerate(row['variant_outputs']),
                            total=len(row['variant_outputs']), leave=False) if not args.interactive else enumerate(row['variant_outputs']):
        if output:
          row['variant_outputs'][i] = amend(output, group_idx)

  utils.save_with_backups(data, args.file)
  print(f'Amended {count} outputs, saved to {args.file}.')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-f', '--file', type=Path, required=True,
                      help='Specify the path to the jsonl file that contains model outputs.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      choices=['input_reasoning', 'output_reasoning'],
                      help='Specify the reasoning type of task.')
  parser.add_argument('-m', '--model', type=str, required=False,
                      help='Specify the model used for inference.')
  parser.add_argument('-i', '--interactive', action='store_true',
                      help='If set, enables interactive mode.')
  args = parser.parse_args()
  match args.task:
    case 'input_reasoning':
      group_idx = 1
    case 'output_reasoning':
      group_idx = 2
    case _:
      raise ValueError(f'Unknown task: {args.task}')

  agent = agent_factory(args.model) if args.model else None
  count = 0
  main()
