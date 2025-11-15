import os
import time
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable

import jsonlines
import torch
from google import genai
from google.genai import types as gtypes
from openai import OpenAI  # type: ignore[attr-defined]
from requests.exceptions import Timeout
from transformers import AutoModelForCausalLM, AutoTokenizer
from io import StringIO

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

  def create_batch_request(self, custom_id: str, sys_prompt: str, user_prompt: str) -> dict:
    """
    Creates a generation request that is compatible with batch API.

    :param custom_id: an unique identifier for the request
    :param sys_prompt: the system prompt
    :param user_prompt: the user prompt
    """
    raise NotImplementedError

  def submit_batch_job(self, display_name: str, requests: list[dict]) -> None:
    """
    Submits generation requests as a batch job to remote platform.

    :param display_name: a name for the created job
    :param requests: a list of generation requests
    """
    raise NotImplementedError

  def retrieve_batch_result(self, job_name: str) -> dict:
    """
    Retrieves the result of a batch job.

    :param job_name: the name of the job
    :return: a mapping from custom ids to generated responses
    """
    raise NotImplementedError


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
  def __init__(self, name: str):
    super().__init__()
    self.client = OpenAI(base_url=os.getenv('BASE_URL'), api_key=os.getenv('API_KEY'))
    if name not in {model.id for model in self.client.models.list()}:
      raise TypeError(f'{name} is not available from {self.client.base_url}.')
    self.name = name

  @retry(retries=setting_dict['agent']['retries'],
         interval=setting_dict['agent']['retry_interval'])
  def generate(self, sys_prompt: str, user_prompt: str) -> str:
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
            'generation_config': {
                'temperature': 0.7,
            },
        },
    }

  def submit_batch_job(self, display_name: str, requests: list[dict]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      batch_file = os.path.join(tmpdir, 'batch.jsonl')
      with jsonlines.open(batch_file, 'w') as writer:
        writer.write_all(requests)
      try:
        uploaded_file = self.client.files.upload(
            file=batch_file,
            config=gtypes.UploadFileConfig(
                display_name=f'batch-requests-{display_name}',
                mime_type='jsonl',
            )
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
          config={
              'display_name': f'batch-job-{display_name}',
          }
      )
    except Exception as e:
      logger.error(f'{e.__class__.__name__} occurred while creating batch job: {e}')
    logger.info(f'Created batch job: {job.name}')

  def retrieve_batch_result(self, job_name: str) -> dict:
    job = self.client.batches.get(name=job_name)
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
    return ''


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
    return ''
