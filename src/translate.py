import re
import subprocess
from typing import NamedTuple, Sequence

import torch
import yaml
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from . import Snippet
from .utils import logger

with open('config/settings.yaml') as f:
  config = yaml.safe_load(f)['translator']


class Translator(NamedTuple):
  name: str
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
  model: any
  gpu: int


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


def load_model(model_name: str, /, gpu_id) -> Translator:
  """
  Load the specified model.
  :param model_name: name of the model
  :return: the model and tokenizer
  """
  logger.info(f'Loading model {model_name}...')

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

  return Translator(model_name, tokenizer, model, gpu_id)


def translate_with_model(snippets: Sequence[Snippet], translator: Translator, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
  """
  Translates snippets in code set with code translation model.
  :param snippets: the snippets to be translated
  :param model: the translation model
  :param tokenizer: the tokenizer
  :param src_lang: source language
  :param dst_lang: destination language
  :return: a set of translated code
  """
  logger.info(f'Translating from {src_lang} to {dst_lang}...')
  # TODO: add API info in dst_language for better accuracy
  prompt = f'{config["prompts"]["prologue"]} \
             <source_language>{src_lang}</source_language> \
             <target_language>{dst_lang}</target_language> \
             <code>```{src_lang}\n%s```</code>'
  translated = [None] * len(snippets)

  for i, snippet in enumerate(snippets):
    torch.cuda.empty_cache()

    match translator.name:
      case 'deepseek-coder-7b-instruct-v1.5' | 'Qwen2.5-Coder-1.5B-Instruct' | 'Qwen2.5-Coder-3B-Instruct' | 'Qwen2.5-Coder-7B-Instruct':
        messages = [{'role': 'user', 'content': prompt % snippet.code}]
        inputs = translator.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
      case 'codegeex2-6b':
        messages = prompt % snippet.code
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
    matched = re.search(r'```\w+\n(.+)```', response, re.DOTALL)
    del inputs, attention_mask, outputs
    if not matched:
      logger.warning(f'Translation of {snippet.id} not found.')
      continue
    logger.info(f'Snippet {i}:\n{matched.group(1)}')
    translated[i] = snippet._replace(code=matched.group(1))
  return translated
