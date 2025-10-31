import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import torch
import yaml
from openai import OpenAI
from requests.exceptions import Timeout
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..logger import logger

with open('configs/settings.yaml') as f:
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
    if name not in {model.id.replace('models/', '') for model in self.client.models.list()}:  # model names from Gemini API have prefix 'models/'
      raise TypeError(f'{name} is not available from {self.client.base_url}.')
    self.name = name

  def generate(self, prompt: Prompt) -> str:
    retry = config['retry']
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
        time.sleep(config['sleep'])
        return completion.choices[0].message.content
      except KeyboardInterrupt:
        logger.warning('Keyboard interrupt.')
        raise
      except Timeout:
        logger.warning(f'Timeout occurred for snippet {prompt.id}. Retrying {attempt + 1}/{retry}...')
        time.sleep(config['retry_interval'])
      except Exception as e:
        logger.error(f'Error occurred for snippet {prompt.id}: {e}...')
        break
    logger.warning(f'Failed to translate snippet {prompt.id} after {retry} attempts.')
    return ''


class DeepseekCoder(BaseAgent):
  @empty_cache
  def __init__(self, name: str):
    root_path = os.getenv("DEEPSEEK_PATH")
    if not root_path:
      raise ValueError('Please set DEEPSEEK_PATH environment variable to the root directory of downloaded DeepSeek Models.')
    path = os.path.join(root_path, name)
    self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True
    )
    if not self.tokenizer.pad_token_id:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  @empty_cache
  def generate(self, prompt: Prompt) -> str:
    messages = [
        {'role': 'system', 'content': prompt.system},
        {'role': 'user', 'content': prompt.user}
    ]
    inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True,
    )
    input_ids = inputs['input_ids'].to(self.model.device)
    attention_mask = inputs['attention_mask'].to(self.model.device)
    with torch.no_grad():
      outputs = self.model.generate(
          input_ids,
          attention_mask=attention_mask,
          max_new_tokens=config['max_new_tokens'],
          do_sample=False,
          temperature=None,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
          pad_token_id=self.tokenizer.pad_token_id,
          use_cache=True,
      )
    response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    del inputs, input_ids, attention_mask, outputs
    return response


class QwenCoder(DeepseekCoder):
  @empty_cache
  def __init__(self, name: str):
    root_path = os.getenv("QWEN_PATH")
    if not root_path:
      raise ValueError('Please set QWEN_PATH environment variable to the root directory of downloaded Qwen Models.')
    path = os.path.join(root_path, name)
    self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )
    if not self.tokenizer.pad_token_id:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class CodeGeeX(BaseAgent):
  @empty_cache
  def __init__(self, name: str):
    path = f'THUDM/{name}'
    self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
    ).eval()
    self.tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )
    if not self.tokenizer.pad_token_id:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  def generate(self, prompt):
    messages = [
        {'role': 'system', 'content': prompt.system},
        {'role': 'user', 'content': prompt.user}
    ]
    inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True,
    )
    input_ids = inputs['input_ids'].to(self.model.device)
    attention_mask = inputs['attention_mask'].to(self.model.device)
    with torch.no_grad():
      outputs = self.model.generate(
          input_ids,
          attention_mask=attention_mask,
          max_new_tokens=config['max_new_tokens'],
          do_sample=False,
          temperature=None,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
          pad_token_id=self.tokenizer.pad_token_id,
          use_cache=True,
      )
    response = self.tokenizer.decode(outputs[:, input_ids.shape[1]:][0], skip_special_tokens=True)
    del inputs, input_ids, attention_mask, outputs
    return response


def agent_factory(name: str) -> BaseAgent:
  """
  Load the specified model.
  :param model_name: name of the model
  :param gpu_id: the GPU id to run locally. If negative, the largest free GPU will be used. If model is remote, this parameter will be ignored.
  :return: the model and tokenizer
  """
  if name.startswith('deepseek-coder'):
    return DeepseekCoder(name)
  if name.startswith('Qwen'):
    return QwenCoder(name)
  if name.startswith('codegeex'):
    return CodeGeeX(name)
  return OpenAIAgent(name)
