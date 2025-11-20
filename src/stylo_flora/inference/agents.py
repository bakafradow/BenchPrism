import os
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from io import StringIO
from typing import ParamSpec
from uuid import uuid4

import jsonlines
import torch
from google import genai
from google.genai import types as gtypes
from openai import OpenAI  # type: ignore[attr-defined]
from requests.exceptions import Timeout
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import setting_dict
from ..logger import logger


class BaseAgent(ABC):
  """
  Abstract base class for LLM-based agents.
  """

  supports_concurrency: bool = True

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

  def create_batch_request(self, custom_id: str, sys_prompt: str, user_prompt: str) -> dict:
    """
    Creates a generation request that is compatible with batch API.

    :param custom_id: an unique identifier for the request
    :param sys_prompt: the system prompt
    :param user_prompt: the user prompt
    """
    raise NotImplementedError

  def submit_batch_job(self, requests: list[dict]) -> None:
    """
    Submits generation requests as a batch to remote platform.

    :param requests: a list of generation requests
    """
    raise NotImplementedError

  def retrieve_batch_result(self, batch_id: str) -> dict:
    """
    Retrieves the result of a batch.

    :param batch_id: the identifier of the batch
    :return: a mapping from custom ids to generated responses
    """
    raise NotImplementedError


P = ParamSpec('P')


def retry(retries: int = 3, interval: float = 30) -> Callable[[Callable[P, str]], Callable[P, str]]:
  def wrapper(func: Callable[P, str]) -> Callable[P, str]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> str:
      for attempt in range(retries):
        try:
          res = func(*args, **kwargs)
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


class OpenAIAgent(BaseAgent):
  supports_concurrency = False

  def __init__(self, name: str):
    super().__init__()
    self.client = OpenAI(base_url=os.getenv('BASE_URL'), api_key=os.getenv('API_KEY'))
    if name not in {model.id for model in self.client.models.list()}:
      raise TypeError(f'{name} is not available from {self.client.base_url}.')
    self.name = name

  @retry(retries=setting_dict['agent']['retries'],
         interval=setting_dict['agent']['retry_interval'])
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
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

  def create_batch_request(self, custom_id: str, sys_prompt: str, user_prompt: str) -> dict:
    return {
        'custom_id': custom_id,
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': self.name,
            'messages': [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        },
    }

  def submit_batch_job(self, requests: list[dict]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      batch_file = os.path.join(tmpdir, f'batch_{uuid4().hex}.jsonl')
      with jsonlines.open(batch_file, 'w') as writer:
        writer.write_all(requests)
      with open(batch_file, 'rb') as f:
        try:
          input_file = self.client.files.create(
              file=f,
              purpose='batch',
          )
        except Exception as e:
          logger.error(f'{e.__class__.__name__} occurred while uploading batch file: {e}')
    logger.info(f'Uploaded {len(requests)} requests from {batch_file}: {input_file.id}')

    try:
      batch = self.client.batches.create(
          input_file_id=input_file.id,
          endpoint='/v1/chat/completions',
          completion_window='24h',
      )
      logger.info(f'Created batch: {batch.id}')
    except Exception as e:
      logger.error(f'{e.__class__.__name__} occurred while creating batch: {e}')

  def retrieve_batch_result(self, batch_id: str) -> dict:
    batch = self.client.batches.retrieve(batch_id)
    match batch.status:
      case 'completed':
        if not batch.output_file_id:
          logger.warning('Result file id is empty.')
          if batch.error_file_id:
            logger.warning(f'Error info:\n{self.client.files.content(batch.error_file_id).text}')
          return {}
        logger.info(f'Downloading result file {batch.output_file_id}...')
        content = self.client.files.content(batch.output_file_id).text
        result = {}
        with jsonlines.Reader(StringIO(content)) as reader:
          for row in reader:
            try:
              candidate = row['response']['body']['choices'][0]
              if candidate['finish_reason'] != 'stop':
                logger.warning(f'Request {row["id"]} (custom_id: {row["custom_id"]}) finished with reason {candidate["finish_reason"]}.')
              result[row['custom_id']] = candidate['message']['content']
              self.token_count += row['response']['body']['usage']['total_tokens']
            except KeyError:
              continue
        return result
      case 'failed':
        logger.info(f'Batch {batch.id} failed. Error: {batch.errors}')
      case 'cancelled':
        logger.info(f'Batch {batch.id} canceled.')
      case 'expired':
        logger.info(f'Batch {batch.id} expired.')
      case _:
        logger.info(f'Batch {batch.id} not completed yet. Status: {batch.status}')
    return {}


class GeminiAgent(BaseAgent):
  supports_concurrency = False

  def __init__(self, name: str):
    super().__init__()
    self.client = genai.Client(
        api_key=os.getenv('API_KEY'),
        http_options=gtypes.HttpOptions(
            timeout=setting_dict['agent']['timeout'] * 1000,
        ),
    )
    if 'models/' + name not in {model.name for model in self.client.models.list()}:
      raise TypeError(f'{name} is not available from Google API.')
    self.name = name

  @retry(retries=setting_dict['agent']['retries'],
         interval=setting_dict['agent']['retry_interval'])
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    res = self.client.models.generate_content(
        model=self.name,
        config=gtypes.GenerateContentConfig(
            system_instruction=sys_prompt,
        ),
        contents=user_prompt,
    )
    time.sleep(setting_dict['agent']['sleep'])
    if res.usage_metadata:
      self.token_count += res.usage_metadata.total_token_count or 0
    return res.text or ''

  def create_batch_request(self, custom_id: str, sys_prompt: str, user_prompt: str) -> dict:
    return {
        'key': custom_id,
        'request': {
            'contents': [{'parts': [{'text': user_prompt}]}],
            'system_instruction': {'parts': [{'text': sys_prompt}]},
        },
    }

  def submit_batch_job(self, requests: list[dict]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      batch_file = os.path.join(tmpdir, 'batch.jsonl')
      with jsonlines.open(batch_file, 'w') as writer:
        writer.write_all(requests)
      try:
        uploaded_file = self.client.files.upload(
            file=batch_file,
            config=gtypes.UploadFileConfig(
                mime_type='jsonl',
            ),
        )
      except Exception as e:
        logger.error(f'{e.__class__.__name__} occurred while uploading batch file: {e}')
    if not uploaded_file.name:
      logger.warning('Name of uploaded file is empty, maybe due to internal error of Google API.')
      return
    logger.info(f'Uploaded {len(requests)} requests from {batch_file}: {uploaded_file.name}')

    try:
      job = self.client.batches.create(
          model=self.name,
          src=uploaded_file.name,
          config=gtypes.CreateBatchJobConfig(),
      )
      logger.info(f'Created batch job: {job.name}')
    except Exception as e:
      logger.error(f'{e.__class__.__name__} occurred while creating batch job: {e}')

  def retrieve_batch_result(self, batch_id: str) -> dict:
    job = self.client.batches.get(name=batch_id)
    match job.state:
      case gtypes.JobState.JOB_STATE_SUCCEEDED:
        if not job.dest or not job.dest.file_name:
          logger.warning('Result file name is empty, maybe due to internal error of Google API.')
          return {}
        result_file_name = job.dest.file_name
        logger.info(f'Downloading result file {result_file_name}...')
        content = self.client.files.download(file=result_file_name).decode('utf-8')
        result = {}
        with jsonlines.Reader(StringIO(content)) as reader:
          for row in reader:
            try:
              candidate = row['response']['candidates'][0]
              if candidate['finishReason'] != 'STOP':
                logger.warning(f'Request {row["key"]} finished with reason {candidate["finishReason"]}.')
              result[row['key']] = candidate['content']['parts'][0]['text']
              self.token_count += row['response']['usageMetadata']['totalTokenCount']
            except KeyError:
              continue
        return result
      case gtypes.JobState.JOB_STATE_FAILED:
        logger.info(f'Batch job {job.name} failed. Error: {job.error}')
      case gtypes.JobState.JOB_STATE_CANCELLED:
        logger.info(f'Batch job {job.name} canceled.')
      case _:
        logger.info(f'Batch job {job.name} not completed yet. State: {job.state}')
    return {}


class Qwen25(BaseAgent):
  def __init__(self, name: str, *, model_path: str | None):
    super().__init__()
    model_path = model_path or f'Qwen/{name}'
    self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if not self.tokenizer.pad_token_id:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  @retry(retries=1)
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True,
    ).to(self.model.device)
    with torch.no_grad():
      outputs = self.model.generate(
          **inputs,
          max_new_tokens=setting_dict['agent']['max_new_tokens'],
          do_sample=False,
          temperature=None,
          top_k=None,
          top_p=None,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
          pad_token_id=self.tokenizer.pad_token_id,
          use_cache=True,
      ).to('cpu')
    response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    del inputs, outputs
    return response


class Phi4(BaseAgent):
  def __init__(self, name: str, *, model_path: str | None):
    super().__init__()
    model_path = model_path or f'microsoft/{name}'
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if not self.tokenizer.pad_token_id:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  @retry(retries=1)
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt',
    ).to(self.model.device)
    with torch.no_grad():
      outputs = self.model.generate(
          inputs,
          max_new_tokens=setting_dict['agent']['max_new_tokens'],
          do_sample=True,
          temperature=.8,
          top_k=50,
          top_p=.95,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
          pad_token_id=self.tokenizer.pad_token_id,
          use_cache=True,
      ).to('cpu')
    response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    del inputs, outputs
    return response


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

  @retry(retries=1)
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True,
    ).to(self.model.device)
    with torch.no_grad():
      outputs = self.model.generate(
          **inputs,
          max_new_tokens=setting_dict['agent']['max_new_tokens'],
          do_sample=False,
          temperature=None,
          top_k=None,
          top_p=None,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
          pad_token_id=self.tokenizer.pad_token_id,
          use_cache=True,
      ).to('cpu')
    response = self.tokenizer.decode(outputs[:, inputs['input_ids'].shape[1]:][0], skip_special_tokens=True)
    del inputs, outputs
    return response


class CodeLlama(BaseAgent):
  def __init__(self, name: str, *, model_path: str | None):
    super().__init__()
    model_path = model_path or f'codellama/{name}'
    self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if not self.tokenizer.pad_token_id:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  @retry(retries=1)
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True,
    ).to(self.model.device)
    with torch.no_grad():
      outputs = self.model.generate(
          **inputs,
          max_new_tokens=setting_dict['agent']['max_new_tokens'],
          do_sample=False,
          temperature=None,
          top_k=None,
          top_p=None,
          num_return_sequences=1,
          eos_token_id=self.tokenizer.eos_token_id,
          pad_token_id=self.tokenizer.pad_token_id,
          use_cache=True,
      ).to('cpu')
    response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    del inputs, outputs
    return response
