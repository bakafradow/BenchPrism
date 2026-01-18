import re
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
from tqdm import tqdm

from scripts.postprocessing import utils
from stylo_flora.inference import agent_factory
from stylo_flora.inference.tasks.tag_classification import CANDIDATES

SYS_PROMPT = f"""
An LLM was instructed to tag a code snippet. It should have responded with a sequence of tags strictly selected from the following candidates separated by commas. However, it failed to do so.
Given its output, your task is to fix its answer. You may turn similar words into exact ones, remove extra text, or adjust the format. Do NOT return anything else. If you REALLY think there is no valid tag in the output, ONLY return one word `N`.

Candidates:
{','.join(CANDIDATES)}

Examples:
1. Given output "Overall, this code is a well-written and efficient solution to a programming contest problem. It uses a combination of dynamic programming, bitmasks, and greedy algorithms to solve the problem in a reasonable amount of time.", you should return "dp,bitmasks,greedy".
2. Given output "DFS and similar,Greedy,brute_force", you should return "dfs and similar,greedy,brute force".
3. Given output "dp,implementation数学相关的动态规划和实现细节是这段代码的核心。", you should return "dp,implementation".
4. Given output "<br/> ", you should return "N".
"""

PATTERN = re.compile(r'^[\w\-/]+(?: [\w\-/]+)*$')


def ok(output: list[str]) -> bool:
  return all(PATTERN.match(tag) for tag in output)


def amend(output: list[str]) -> list[str]:
  global count
  if ok(output):
    return output
  res = agent.generate(sys_prompt=SYS_PROMPT, user_prompt=','.join(output))
  if not res:
    print(f'Failed to amend output:\n{output}')
    return output
  res = res.strip()
  if res.strip() == 'N':
    return ['N']
  if ok(amended := res.split(',')):
    count += 1
    return amended
  print(f'Failed to resolve response for {output}: {repr(res)}')
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
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model used for inference.')
  args = parser.parse_args()

  agent = agent_factory(args.model)
  count = 0
  main()
