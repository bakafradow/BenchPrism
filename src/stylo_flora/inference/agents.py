import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch
import yaml
from google import genai
from openai import OpenAI  # type: ignore[attr-defined]
from requests.exceptions import Timeout
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..logger import logger

with open('configs/settings.yaml') as f:
  config = yaml.safe_load(f)['agent']


class Prompt(NamedTuple):
  id: str
  system: str
  user: str


class BaseAgent(ABC):
  """
  Abstract base class for LLM-based agents.
  """

  def __init__(self):
    self.token_count = 0

  @abstractmethod
  def generate(self, prompt: Prompt) -> str:
    """
    Generates a response based on the provided prompt.
    :param prompt: the input prompt for the model
    :return: the generated response
    """
    pass


def retry(retries: int, interval: float) -> Callable:
  def wrapper(func: Callable) -> Callable:
    def inner(self, prompt: Prompt) -> str:
      for attempt in range(retries):
        try:
          res = func(self, prompt)
          return res
        except KeyboardInterrupt:
          logger.warning('Keyboard interrupt.')
          raise
        except Timeout:
          logger.warning(f'Timeout occurred for snippet {prompt.id}. Retrying {attempt + 1}/{retries}...')
          time.sleep(interval)
        except Exception as e:
          logger.error(f'{e.__class__.__name__} occurred for snippet {prompt.id}: {e}...')
          break
      logger.warning(f'Failed to generate for snippet {prompt.id} after {retries} attempts.')
      return ''
    return inner
  return wrapper


def empty_cache(func: Callable) -> Callable:
  def wrapper(*args, **kwargs):
    torch.cuda.empty_cache()
    return func(*args, **kwargs)
  return wrapper


# TODO 3: support more local models
class OpenAIAgent(BaseAgent):
  def __init__(self, name: str):
    super().__init__()
    self.client = OpenAI(base_url=os.getenv('BASE_URL'), api_key=os.getenv('API_KEY'))
    if name not in {model.id for model in self.client.models.list()}:
      raise TypeError(f'{name} is not available from {self.client.base_url}.')
    self.name = name

  @retry(retries=config['retries'], interval=config['retry_interval'])
  def generate(self, prompt: Prompt) -> str:
    completion = self.client.chat.completions.create(
        model=self.name,
        messages=[
            {'role': 'system', 'content': prompt.system},
            {'role': 'user', 'content': prompt.user}
        ],
        timeout=config['timeout'],
    )
    if completion.usage:
      self.token_count += completion.usage.total_tokens
    time.sleep(config['sleep'])
    return completion.choices[0].message.content or ''


class GeminiAgent(BaseAgent):
  def __init__(self, name: str):
    super().__init__()
    self.client = genai.Client(
        api_key=os.getenv('API_KEY'),
        http_options=genai.types.HttpOptions(
            timeout=config['timeout'] * 1000,
        ),
    )
    if 'models/' + name not in {model.name for model in self.client.models.list()}:
      raise TypeError(f'{name} is not available from Google API.')
    self.name = name

  @retry(retries=config['retries'], interval=config['retry_interval'])
  def generate(self, prompt: Prompt) -> str:
    res = self.client.models.generate_content(
        model=self.name,
        config=genai.types.GenerateContentConfig(
            system_instruction=prompt.system,
        ),
        contents=prompt.user,
    )
    time.sleep(config['sleep'])
    if res.usage_metadata:
      self.token_count += res.usage_metadata.total_token_count or 0
    return res.text or ''


class Qwen25(BaseAgent):
  def __init__(self, name: str, model_dir: Path | None):
    super().__init__()
    if not model_dir:
      raise ValueError('Please provide path to the Qwen 2.5 model through --model-dir argument.')
    self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
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


class Phi4(BaseAgent):
  def __init__(self, name: str, model_dir: Path | None):
    super().__init__()

  @empty_cache
  def generate(self, prompt: Prompt) -> str:
    return ''


class CodeGeeX4(BaseAgent):
  def __init__(self, name: str, model_dir: Path | None):
    super().__init__()
    path = model_dir if model_dir else f'THUDM/{name}'
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

  @empty_cache
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


class CodeLlama(BaseAgent):
  def __init__(self, name: str, model_dir: Path | None):
    super().__init__()

  @empty_cache
  def generate(self, prompt: Prompt) -> str:
    return ''


def agent_factory(name: str, model_dir: Path | None) -> BaseAgent:
  """
  Load the specified model.
  :param name: name of the model
  :param model_dir: directory to load the model
  :return: an encapsulated agent instance
  """
  if 'gemini' in name:
    return GeminiAgent(name)
  if 'qwen2.5' in name:
    return Qwen25(name, model_dir)
  if 'phi-4' in name:
    return Phi4(name, model_dir)
  if 'codegeex4' in name:
    return CodeGeeX4(name, model_dir)
  if 'codellama' in name:
    return CodeLlama(name, model_dir)
  return OpenAIAgent(name)
