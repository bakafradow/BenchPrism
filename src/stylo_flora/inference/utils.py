import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml
from tqdm import tqdm

from .. import Snippet
from ..logger import logger

with open('settings.yml') as f:
  config = yaml.safe_load(f)['agent']


def get_freest_gpu() -> str:
  """
  Finds the GPU with the largest available memory, or the CPU if no GPU is available.
  :return: the freest CUDA device or 'cpu'
  """
  if not torch.cuda.is_available():
    logger.warning('No CUDA devices available. Using CPU instead.')
    return 'cpu'
  gpu_count = torch.cuda.device_count()
  if gpu_count == 0:
    logger.warning('Current device has 0 GPUs. Using CPU instead.')
    return 'cpu'

  returned = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
  memories = [line.split(', ') for line in returned.stdout.decode('utf-8').strip().split('\n')]
  free_memories = [int(total) - int(used) for total, used in memories]
  gpu = max(range(gpu_count), key=lambda i: free_memories[i])
  logger.info(f'Using GPU {gpu} with {free_memories[gpu]} MB free memory.')
  return f'cuda:{gpu}'


def work(worker: Callable[[int, Snippet], Snippet | None], snippets: Sequence[Snippet]) -> Sequence[Snippet]:
  max_workers = max(1, config['max_workers'])
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    return tuple(tqdm(executor.map(worker, range(len(snippets)), snippets),
                      desc='Generating', total=len(snippets), leave=False))
