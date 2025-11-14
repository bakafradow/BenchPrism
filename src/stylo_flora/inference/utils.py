import subprocess
from typing import Any

import torch

from .. import Snippet
from ..logger import logger
from .agents import BaseAgent
from .tasks.base import BaseTask


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


def task_worker(
    agent: BaseAgent,
    task: BaseTask,
    snippet: Snippet | None,
) -> Any:
  if not snippet:
    return None
  sys_prompt, user_prompt = task.get_prompt(snippet)
  res = agent.generate(sys_prompt, user_prompt)
  if not res:
    return None
  res = task.resolve_response(res)
  if not res:
    logger.warning(f'Failed to resolve response for snippet {snippet.id}')
  return res
