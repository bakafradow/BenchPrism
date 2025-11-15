import re
import time
from argparse import ArgumentParser, Namespace
from operator import itemgetter
from pathlib import Path

import jsonlines
from tqdm import tqdm

from stylo_flora.inference import agent_factory, task_factory
from stylo_flora.logger import init_logger, logger


def parse_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model used for inference.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      choices=[
                          'code_translation',
                          'code_repair',
                          'code2tag',
                          'descode2tag',
                          'code_summarization',
                          'input_reasoning',
                          'output_reasoning',
                          'mcq_answering',
                          'test_generation',
                      ],
                      help='Specify the code task to evaluate on.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('-j', '--job', type=str, required=True,
                      help='Specify the job name returned by the API platform.')
  parser.add_argument('-f', '--file', type=Path, required=True,
                      help='Specify the path to the jsonl file that contains model outputs.')
  parser.add_argument('--retry-interval', type=int, required=False, default=30,
                      help='Specify retry interval (in seconds) in case the job is not ready.')
  args = parser.parse_args()
  return args


def main():
  args = parse_args()
  init_logger(path=None, verbose=False, debug=False)
  logger.info(f'Initializing model {args.model}...')
  agent = agent_factory(name=args.model)
  logger.info(f'Initializing task {args.task}...')
  task = task_factory(args.task, **dict(args._get_kwargs()))

  while True:
    result = agent.retrieve_batch_result(args.job)
    if result:
      logger.info(f'Retrieved {len(result)} outputs from {args.job}, {agent.token_count} tokens used in total.')
      break
    for _ in tqdm(range(args.retry_interval), desc='Retry after',
                  leave=False, unit='s', bar_format='{l_bar}{bar}'):
      time.sleep(1)

  with jsonlines.open(args.file, mode='r') as reader:
    data = {row['id']: row for row in reader}

  for k, v in result.items():
    res = task.resolve_response(v)
    if not res:
      continue

    matched = re.match(r'^(.+)_(orig|\d+)$', k)
    if not matched or len(matched.groups()) != 2:
      logger.error(f'Invalid key: {k}')
      continue
    snippet_id, suffix = matched.groups()

    # data dict should be initialized with `None`s during the submission of batch job
    if suffix == 'orig':
      data[snippet_id]['output'] = res
    else:
      seq = int(suffix)
      data[snippet_id]['variant_outputs'][seq] = res

  with jsonlines.open(args.file, mode='w') as writer:
    for row in sorted(data.values(), key=itemgetter('id')):
      writer.write(row)
  logger.info(f'Saved outputs to {args.file} with {len(data)} rows.')


if __name__ == '__main__':
  main()
