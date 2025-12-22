import subprocess
from typing import Any

import torch

from .. import Snippet
from ..logger import logger
from .agents import BaseAgent, GeminiAgent, LocalAgent, OpenAIAgent, ZhipuAgent
from .tasks import (BaseTask, CodeRepair, CodeRepairUJB, CodeSummarization,
                    CodeTranslation, DefectDetectionUJB, IOReasoning,
                    MCQAnswering, TagClassification, TestGeneration,
                    TestGenerationTB)
from .tasks.io_reasoning import ReasoningType


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

  completed = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
  memories = [line.split(', ') for line in completed.stdout.decode('utf-8').strip().split('\n')]
  free_memories = [int(total) - int(used) for total, used in memories]
  gpu = max(range(gpu_count), key=lambda i: free_memories[i])
  logger.info(f'Using GPU {gpu} with {free_memories[gpu]} MB free memory.')
  return f'cuda:{gpu}'


def agent_factory(name: str) -> BaseAgent:
  """
  Load the specified model.
  :param name: name of the model
  :param model_path: path to the local model directory, only for open-source models
  :return: an encapsulated agent instance
  """
  parts = name.split(':', 1)
  if len(parts) == 1:
    return LocalAgent(name)
  platform, model = parts
  match platform.lower():
    case 'openai':
      return OpenAIAgent(model)
    case 'gemini':
      return GeminiAgent(model)
    case 'zhipu':
      return ZhipuAgent(model)
    case _:
      raise ValueError(f'Unsupported platform {platform}.')


def task_factory(name: str, **kwargs) -> BaseTask:
  if kwargs['dataset'].lower() == 'coderujb':
    match name:
      case 'code_repair':
        return CodeRepairUJB(kwargs['src_lang'])
      case 'defect_detection':
        return DefectDetectionUJB(kwargs['src_lang'])
      case _:
        raise ValueError(f'Unknown task on CoderUJB: {name}')

  if kwargs['dataset'].lower() == 'testbench':
    match name:
      case 'test_generation':
        return TestGenerationTB(kwargs['src_lang'])
      case _:
        raise ValueError(f'Unknown task on TestBench: {name}')

  match name:
    case 'code_translation':
      return CodeTranslation(kwargs['src_lang'], kwargs['dst_lang'])
    case 'code_repair':
      return CodeRepair(kwargs['src_lang'])
    case 'code2tag':
      return TagClassification(kwargs['src_lang'], with_desc=False)
    case 'descode2tag':
      return TagClassification(kwargs['src_lang'], with_desc=True)
    case 'test_generation':
      return TestGeneration(kwargs['src_lang'])
    case 'code_summarization':
      return CodeSummarization(kwargs['src_lang'])
    case 'mcq_answering':
      return MCQAnswering(kwargs['src_lang'])
    case 'input_reasoning':
      return IOReasoning(kwargs['src_lang'], ReasoningType.INPUT_REASONING)
    case 'output_reasoning':
      return IOReasoning(kwargs['src_lang'], ReasoningType.OUTPUT_REASONING)
    case _:
      raise ValueError(f'Unknown task: {name}')


def task_worker(
    agent: BaseAgent,
    task: BaseTask,
    snippet: Snippet | None,
) -> Any:
  if not snippet:
    return None
  prompts = task.get_prompt(snippet)
  if not prompts:
    return None
  res = agent.generate(*prompts)
  if not res:
    return None
  res = task.resolve_response(res)
  if not res:
    logger.warning(f'Failed to resolve response for snippet {snippet.id}')
  return res
