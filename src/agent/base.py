import os
import time
from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import torch
import yaml
from openai import OpenAI
from requests.exceptions import Timeout
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..logger import logger
from .utils import get_freest_gpu

with open('settings.yml') as f:
  config = yaml.safe_load(f)['agent']


class Prompt(NamedTuple):
  id: str
  system: str
  user: str


def empty_cache(func: Callable):
  def wrapper(*args, **kwargs):
    torch.cuda.empty_cache()
    return func(*args, **kwargs)
  return wrapper


class BaseAgent(ABC):
  """
  Abstract base class for LLM-based agents.
  """

  @abstractmethod
  def generate(self, prompt: Prompt) -> str:
    """
    Generates a response based on the provided prompt.
    :param prompt: the input prompt for the model
    :return: the generated response
    """
    pass


class OpenAIAgent(BaseAgent):
  def __init__(self, name: str):
    self.client = OpenAI(base_url=os.getenv('BASE_URL'), api_key=os.getenv('API_KEY'))
    if name not in {model.id for model in self.client.models.list()}:
      raise TypeError(f'{name} is not available from {self.client.base_url}.')
    self.name = name

  def generate(self, prompt: Prompt) -> str:
    retry = config['retry']
    retry_interval = config['retry_interval']
    for attempt in range(retry):
      try:
        completion = self.client.chat.completions.create(
            model=self.name,
            messages=[
                {'role': 'system', 'content': prompt.system},
                {'role': 'user', 'content': prompt.user}
            ],
            timeout=config['timeout'],
        )
        return completion.choices[0].message.content
      except KeyboardInterrupt:
        logger.warning('Keyboard interrupt.')
        raise
      except Timeout:
        logger.warning(f'Timeout occurred for snippet {prompt.id}. Retrying {attempt + 1}/{retry}...')
        time.sleep(retry_interval)
      except Exception as e:
        logger.error(f'Error occurred for snippet {prompt.id}: {e}...')
        break
    logger.warning(f'Failed to translate snippet {prompt.id} after {retry} attempts.')
    return ''


class DeepseekCoder(BaseAgent):
  @empty_cache
  def __init__(self, name: str, device: str):
    path = os.getenv("DEEPSEEKCODER_PATH")
    if not path:
      raise ValueError('DEEPSEEKCODER_PATH environment variable is not set.')
    model_args = {
        'pretrained_model_name_or_path': os.path.join(path, name),
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16,
        'device_map': 'auto',
    }
    self.model = AutoModelForCausalLM.from_pretrained(**model_args).cuda()
    self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

  @empty_cache
  def generate(self, prompt: Prompt) -> str:
    messages = [
        {'role': 'system', 'content': prompt.system},
        {'role': 'user', 'content': prompt.user}
    ]
    inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(self.model.device)
    outputs = self.model.generate(
        inputs,
        max_new_tokens=config['max_new_tokens'],
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=self.tokenizer.eos_token_id,
        use_cache=True,
    )
    response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    del inputs, outputs
    return response


class QwenCoder(DeepseekCoder):
  @empty_cache
  def __init__(self, name: str, device: str):
    path = os.getenv("QWENCODER_PATH")
    if not path:
      raise ValueError('QWENCODER_PATH environment variable is not set.')
    model_args = {
        'pretrained_model_name_or_path': os.path.join(path, name),
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16,
        'device_map': 'auto',
    }
    self.model = AutoModelForCausalLM.from_pretrained(**model_args).cuda()
    self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


class CodeGeeX(BaseAgent):
  @empty_cache
  def __init__(self, name: str, device: str):
    model_args = {
        'pretrained_model_name_or_path': f'THUDM/{name}',
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16,
        'low_cpu_mem_usage': True,
        'device_map': 'auto',
    }
    self.model = AutoModelForCausalLM.from_pretrained(**model_args).cuda().eval()
    self.tokenizer = AutoTokenizer.from_pretrained(f'THUDM/{name}', trust_remote_code=True)

  @empty_cache
  def generate(self, prompt: Prompt) -> str:
    messages = [
        {'role': 'system', 'content': prompt.system},
        {'role': 'user', 'content': prompt.user}
    ]
    inputs = self.tokenizer.encode(messages, add_generation_prompt=True, tokenize=True, return_tensors='pt', return_dict=True).to(self.model.device)
    with torch.no_grad():
      outputs = self.model.generate(
          inputs,
          max_new_tokens=config['max_new_tokens'],
          do_sample=False,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
          pad_token_id=self.tokenizer.pad_token_id,
          use_cache=True,
      )
    response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    del inputs, outputs
    return response


def agent_factory(name: str, device: str = 'auto') -> BaseAgent:
  """
  Load the specified model.
  :param model_name: name of the model
  :param gpu_id: the GPU id to run locally. If negative, the largest free GPU will be used. If model is remote, this parameter will be ignored.
  :return: the model and tokenizer
  """
  device = get_freest_gpu() if device == 'auto' else device
  if name.startswith('deepseek-coder'):
    return DeepseekCoder(name, device)
  if name.startswith('Qwen'):
    return QwenCoder(name, device)
  if name.startswith('codegeex'):
    return CodeGeeX(name, device)
  return OpenAIAgent(name)
