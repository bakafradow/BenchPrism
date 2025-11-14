import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from google import genai
from openai import OpenAI  # type: ignore[attr-defined]
from requests.exceptions import Timeout
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import setting_dict
from ..logger import logger


class BaseAgent(ABC):
  """
  Abstract base class for LLM-based agents.
  """

  def __init__(self):
    self.token_count = 0

  @abstractmethod
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    """
    Generates a response based on the provided prompt.
    :param sys_prompt: the system prompt
    :param user_prompt: the user prompt
    :return: the generated response
    """
    pass


def retry(retries: int, interval: float) -> Callable:
  def wrapper(func: Callable) -> Callable:
    def inner(self, sys_prompt: str, user_prompt: str) -> str:
      for attempt in range(retries):
        try:
          res = func(self, sys_prompt, user_prompt)
          return res
        except KeyboardInterrupt:
          logger.warning('Keyboard interrupt.')
          raise
        except Timeout:
          logger.warning(f'Timeout. Retrying {attempt + 1}/{retries}...')
          time.sleep(interval)
        except Exception as e:
          logger.error(f'{e.__class__.__name__} occurred: {e}...')
          break
      logger.warning(f'Failed to generate after {retries} attempts.')
      return ''
    return inner
  return wrapper


def empty_cache(func: Callable) -> Callable:
  def wrapper(*args, **kwargs):
    torch.cuda.empty_cache()
    return func(*args, **kwargs)
  return wrapper


class OpenAIAgent(BaseAgent):
  def __init__(self, name: str, *, batch_api: bool):
    super().__init__()
    self.client = OpenAI(base_url=os.getenv('BASE_URL'), api_key=os.getenv('API_KEY'))
    if name not in {model.id for model in self.client.models.list()}:
      raise TypeError(f'{name} is not available from {self.client.base_url}.')
    self.name = name
    self.batch_api = batch_api

  @retry(retries=setting_dict['agent']['retries'],
         interval=setting_dict['agent']['retry_interval'])
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    # TODO 1: use batch API to save tokens
    # TODO 2: choose representative hyperparameters
    completion = self.client.chat.completions.create(
        model=self.name,
        messages=[
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        timeout=setting_dict['agent']['timeout'],
    )
    if completion.usage:
      self.token_count += completion.usage.total_tokens
    time.sleep(setting_dict['agent']['sleep'])
    return completion.choices[0].message.content or ''


class GeminiAgent(BaseAgent):
  def __init__(self, name: str, *, batch_api: bool):
    super().__init__()
    self.client = genai.Client(
        api_key=os.getenv('API_KEY'),
        http_options=genai.types.HttpOptions(
            timeout=setting_dict['agent']['timeout'] * 1000,
        ),
    )
    if 'models/' + name not in {model.name for model in self.client.models.list()}:
      raise TypeError(f'{name} is not available from Google API.')
    self.name = name
    self.batch_api = batch_api

  @retry(retries=setting_dict['agent']['retries'],
         interval=setting_dict['agent']['retry_interval'])
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    res = self.client.models.generate_content(
        model=self.name,
        config=genai.types.GenerateContentConfig(
            system_instruction=sys_prompt,
        ),
        contents=user_prompt,
    )
    time.sleep(setting_dict['agent']['sleep'])
    if res.usage_metadata:
      self.token_count += res.usage_metadata.total_token_count or 0
    return res.text or ''


class Qwen25(BaseAgent):
  def __init__(self, name: str, *, model_path: str | None):
    super().__init__()
    model_path = model_path or f'Qwen/{name}'
    self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if not self.tokenizer.pad_token_id:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  @empty_cache
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt}
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
          max_new_tokens=setting_dict['agent']['max_new_tokens'],
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
  def __init__(self, name: str, *, model_path: str | None):
    super().__init__()
    model_path = model_path or f'microsoft/{name}'

  @empty_cache
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    return ''  # TODO


class CodeGeeX4(BaseAgent):
  def __init__(self, name: str, *, model_path: str | None):
    super().__init__()
    model_path = model_path or f'THUDM/{name}'
    self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
    ).eval()
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if not self.tokenizer.pad_token_id:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  @empty_cache
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt}
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
          max_new_tokens=setting_dict['agent']['max_new_tokens'],
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
  def __init__(self, name: str, *, model_path: str | None):
    super().__init__()
    model_path = model_path or f'meta-llama/{name}'

  @empty_cache
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    return ''  # TODO


def agent_factory(name: str, *, batch_api: bool = False, model_path: str | None = None) -> BaseAgent:
  """
  Load the specified model.
  :param name: name of the model
  :param model_path: path to the local model directory, only for open-source models
  :param batch_api: whether to use batch API to generate, only for proprietary models
  :return: an encapsulated agent instance
  """
  if 'gemini' in name:
    return GeminiAgent(name, batch_api=batch_api)
  if 'qwen2.5' in name:
    return Qwen25(name, model_path=model_path)
  if 'phi-4' in name:
    return Phi4(name, model_path=model_path)
  if 'codegeex4' in name:
    return CodeGeeX4(name, model_path=model_path)
  if 'codellama' in name:
    return CodeLlama(name, model_path=model_path)
  return OpenAIAgent(name, batch_api=batch_api)  # default to OpenAI-compatible agents
