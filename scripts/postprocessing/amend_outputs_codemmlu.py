from argparse import ArgumentParser
from pathlib import Path

import jsonlines
from tqdm import tqdm

from scripts.postprocessing import utils
from stylo_flora.inference import agent_factory

SYS_PROMPT = """
An LLM was instructed to answer a choice question. It should have responded with a single letter, A, B, C or D. However, it failed to do so.
Given its output, your task is to recognize and extract its answer. Do NOT return anything else. If you think that part doesn't exist in the output, ONLY return `N`.

Examples:
1. Given output "The most probable behavior is a Compile Error.\n\nHere's why: ...\n\nTherefore, ...\n\nThe final answer is $\\boxed{A}$", you should return "A".
2. Given output "The most probable behavior is a Runtime Error.", you should return "N".
"""

def amend(output: str) -> str:
  global count
  if len(output) == 1:
    return output
  res = agent.generate(sys_prompt=SYS_PROMPT, user_prompt=output)
  if res and res.strip() in ['A', 'B', 'C', 'D']:
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
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model used for inference.')
  args = parser.parse_args()

  agent = agent_factory(args.model)
  count = 0
  main()
