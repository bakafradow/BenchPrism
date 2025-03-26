import re
import subprocess
import time
from typing import NamedTuple, Sequence

import torch
import yaml
from openai import OpenAI
from requests.exceptions import Timeout
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from . import Snippet
from .utils import logger

LOCAL_MODELS = ['deepseek-coder-7b-instruct-v1.5', 'Qwen2.5-Coder-1.5B-Instruct', 'Qwen2.5-Coder-3B-Instruct', 'Qwen2.5-Coder-7B-Instruct', 'codegeex2-6b']

with open('config/settings.yaml') as f:
  config = yaml.safe_load(f)['translator']


class Translator(NamedTuple):
  """
  A translator that wraps the components of a translation model. If the model is non-local, the tokenizer and gpu won't be used.
  """
  name: str
  model: any
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None
  gpu: int = -1


def _get_largest_free_gpu() -> int:
  """
  Finds the GPU with the largest available memory.
  Returns:
      int: The index of the GPU with the largest free memory, or None if no GPU is available.
  """
  if not torch.cuda.is_available():
    raise ValueError('No GPU available.')
  gpu_count = torch.cuda.device_count()
  if gpu_count == 0:
    raise ValueError('Current device has 0 GPUs.')

  returned = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
  memories = [line.split(', ') for line in returned.stdout.decode('utf-8').strip().split('\n')]
  free_memories = [int(total) - int(used) for total, used in memories]
  return max(range(gpu_count), key=lambda i: free_memories[i])


# FIXME: reduce stamp coupling
def load_model(model_name: str, *, gpu_id: int = -1) -> Translator:
  """
  Load the specified model.
  :param model_name: name of the model
  :param gpu_id: the GPU id to run locally. If negative, the largest free GPU will be used. If model is remote, this parameter will be ignored.
  :return: the model and tokenizer
  """
  logger.info(f'Loading model {model_name}...')

  if model_name not in LOCAL_MODELS:
    client = OpenAI(base_url=config['base_url'], api_key=config['api_key'])
    if model_name not in {model.id for model in client.models.list()}:
      raise ValueError(f'{model_name} is not available.')
    return Translator(model_name, client)

  torch.cuda.empty_cache()
  if gpu_id < 0:
    gpu_id = _get_largest_free_gpu()
  quantization_config = BitsAndBytesConfig(load_in_8bit=True)
  model_args = {
      'trust_remote_code': True,
      'torch_dtype': torch.float16,
      'device_map': f'cuda:{gpu_id}',
      'quantization_config': quantization_config,
  }

  match model_name:
    case 'deepseek-coder-7b-instruct-v1.5' | 'Qwen2.5-Coder-1.5B-Instruct' | 'Qwen2.5-Coder-3B-Instruct' | 'Qwen2.5-Coder-7B-Instruct':
      tokenizer = AutoTokenizer.from_pretrained(config['models'][model_name], trust_remote_code=True)
      model = AutoModelForCausalLM.from_pretrained(config['models'][model_name], **model_args)
    case 'codegeex2-6b':
      tokenizer = AutoTokenizer.from_pretrained(f'THUDM/{model_name}', trust_remote_code=True)
      model = AutoModel.from_pretrained(f'THUDM/{model_name}', trust_remote_code=True, torch_dtype=torch.float16, device=f'cuda:{gpu_id}')
      model = model.eval()
    case _:
      raise TypeError(f'{model_name} is unsupported yet.')

  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

  return Translator(model_name, model, tokenizer, gpu_id)


class PromptPair(NamedTuple):
  system: str
  user: str


def _build_prompt(src_lang: str, dst_lang: str) -> PromptPair:
  system = config['prompts']['prologue']
  user = f'<source_language>{src_lang}</source_language> \
            <target_language>{dst_lang}</target_language> \
            <source_code>```{src_lang}\n%s```</source_code> \
            <target_declaration>```{dst_lang}\n%s```</target_declaration>'
  return PromptPair(system, user)


def _translate_remotely(translator: Translator, snippet: Snippet, prompt: PromptPair) -> str:
  retry = config['retry']
  retry_interval = config['retry_interval']
  for attempt in range(retry):
    try:
      completion = translator.model.chat.completions.create(
        model=translator.name,
        messages=[
          {'role': 'system', 'content': prompt.system},
          {'role': 'user', 'content': prompt.user % (snippet.code, snippet.ref)}
        ],
        timeout=120,
      )
      return completion.choices[0].message.content
    except Timeout:
      logger.warning(f'Timeout occurred for snippet {snippet.id}. Retrying {attempt + 1}/{retry}...')
      time.sleep(retry_interval)
    except Exception as e:
      logger.error(f'Error occurred for snippet {snippet.id}: {e}...')
      break
  logger.warning(f'Failed to translate snippet {snippet.id} after {retry} attempts.')


def _translate_locally(translator: Translator, snippet: Snippet, prompt: PromptPair) -> str:
  torch.cuda.empty_cache()

  match translator.name:
    case 'deepseek-coder-7b-instruct-v1.5' | 'Qwen2.5-Coder-1.5B-Instruct' | 'Qwen2.5-Coder-3B-Instruct' | 'Qwen2.5-Coder-7B-Instruct':
      messages=[
        {'role': 'system', 'content': prompt.system},
        {'role': 'user', 'content': prompt.user % (snippet.code, snippet.ref)}
      ]
      inputs = translator.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
    case 'codegeex2-6b':
      messages = prompt % (snippet.code, snippet.ref)
      inputs = translator.tokenizer.encode(messages, return_tensors='pt', padding=True)
    case _:
      raise TypeError(f'{translator.name} is unsupported yet.')

  inputs = inputs.to(f'cuda:{translator.gpu}')
  attention_mask = torch.ones_like(inputs).to(f'cuda:{translator.gpu}')
  outputs = translator.model.generate(
      inputs,
      attention_mask=attention_mask,
      max_new_tokens=config['max_new_tokens'],
      do_sample=False,
      num_return_sequences=1,
      eos_token_id=translator.tokenizer.eos_token_id,
      pad_token_id=translator.tokenizer.pad_token_id,
      use_cache=True,
  )
  response = translator.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
  del inputs, attention_mask, outputs
  return response


def translate_with_model(translator: Translator, snippets: Sequence[Snippet], src_lang: str, dst_lang: str) -> Sequence[Snippet]:
  """
  Translates snippets in code set with code translation model.
  :param translator: the translation model
  :param snippets: the snippets to be translated
  :param dataset: dataset name
  :param src_lang: source language
  :param dst_lang: destination language
  :return: a sequence of translated code
  """
  logger.info(f'Translating from {src_lang} to {dst_lang}...')
  translated = [None] * len(snippets)
  prompt = _build_prompt(src_lang, dst_lang)
  for i, snippet in enumerate(snippets):
    if translator.name not in LOCAL_MODELS:
      response = _translate_remotely(translator, snippet, prompt)
    else:
      response = _translate_locally(translator, snippet, prompt)
    matched = re.search(r'```\w+\n(.+)```', response, re.DOTALL)
    if not matched:
      logger.warning(f'Translation of {snippet.id} not found.')
      logger.verbose(response)
      continue
    logger.verbose(f'Snippet {i}:\n{matched.group(1)}')
    translated[i] = snippet._replace(code=matched.group(1))
  return translated
