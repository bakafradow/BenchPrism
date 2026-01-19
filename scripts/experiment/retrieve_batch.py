import os
import re
import shutil
import subprocess
import time
from argparse import ArgumentParser, Namespace
from operator import itemgetter
from pathlib import Path

import jsonlines
from tqdm import tqdm

from scripts.experiment.run import SUPPORTED_TASKS
from scripts.experiment.utils import get_eval_id
from benchprism.inference import agent_factory, task_factory
from benchprism.logger import init_logger, logger


def parse_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model used for inference.')
  parser.add_argument('-t', '--task', type=str, required=True,
                      choices=SUPPORTED_TASKS,
                      help='Specify the code task to evaluate on.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  parser.add_argument('--dst-lang', type=str, required=False,
                      help='Specify the destination language. Only used for code translation task.')
  parser.add_argument('--result-dir', type=Path, required=True,
                      help='Directory to save the results.')
  parser.add_argument('--seed', type=int, default=42,
                      help='Set the random seed for reproducibility.')
  parser.add_argument('-j', '--job', type=str, required=True,
                      help='Specify the job name returned by the API platform.')
  parser.add_argument('--retry-interval', type=int, required=False, default=60,
                      help='Specify retry interval (in seconds) in case the job is not ready.')
  args = parser.parse_args()
  eval_id = get_eval_id(args)
  args.outputs_path = args.result_dir / f'outputs_{eval_id}.jsonl'
  return args


def _retrieve() -> dict:
  while True:
    result = agent.retrieve_batch_result(args.job)
    if result:
      info = f'Retrieved {len(result)} outputs from {args.job}, {agent.token_count} tokens used in total.'
      logger.info(info)
      _message(info)
      return result
    elif result is not None:
      logger.info('The batch job stopped without results. Stop retrying.')
      exit(1)
    try:
      for _ in tqdm(range(args.retry_interval), desc='Retry after',
                    leave=False, unit='s', bar_format='{l_bar}{bar}'):
        time.sleep(1)
    except KeyboardInterrupt:
      exit(0)


def _message(text: str) -> None:
  if not shutil.which('powershell.exe'):
    return
  cmd = (
      f'Add-Type -AssemblyName PresentationFramework;'
      f'[System.Windows.MessageBox]::Show("{text}", "{os.path.basename(__file__)}", '
      f'0, 64, 0, 0x00020000)'
  )
  subprocess.run(['powershell.exe', '-Command', cmd], capture_output=True)


def main():
  result = _retrieve()

  with jsonlines.open(args.outputs_path, mode='r') as reader:
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
    try:
      if suffix == 'orig':
        data[snippet_id]['output'] = res
      else:
        seq = int(suffix)
        data[snippet_id]['variant_outputs'][seq] = res
    except KeyError as e:
      logger.error(f'Invalid key {e}. Please make sure the batch corresponds to correct experiment!')
      exit(1)

  with jsonlines.open(args.outputs_path, mode='w') as writer:
    writer.write_all(sorted(data.values(), key=itemgetter('id')))
  logger.info(f'Saved outputs to {args.outputs_path} with {len(data)} rows.')


if __name__ == '__main__':
  args = parse_args()
  init_logger()
  logger.info(f'Initializing model {args.model}...')
  agent = agent_factory(name=args.model)
  logger.info(f'Initializing task {args.task}...')
  task = task_factory(args.task, **dict(args._get_kwargs()))

  main()
