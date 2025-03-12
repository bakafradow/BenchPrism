from typing import NamedTuple, Sequence

import re
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, BitsAndBytesConfig

from . import Snippet

with open('config/settings.yaml') as f:
  config = yaml.safe_load(f)['translator']


class Translator(NamedTuple):
  name: str
  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
  model: any


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

    largest_free_gpu_index = 0
    largest_free_memory = 0

    for i in range(gpu_count):
        try:
            # Get device properties
            properties = torch.cuda.get_device_properties(i)
            # Get total memory
            total_memory = properties.total_memory
            # Get memory allocated by pytorch
            allocated_memory = torch.cuda.memory_allocated(i)
            # Calculate free memory
            free_memory = total_memory - allocated_memory

            if free_memory > largest_free_memory:
                largest_free_memory = free_memory
                largest_free_gpu_index = i
        except Exception as e:
            print(f"Error checking GPU {i}: {e}")

    return largest_free_gpu_index


def load_model(model_name: str) -> Translator:
  """
  Load the specified model.
  :param model_name: name of the model
  :return: the model and tokenizer
  """
  print(f'Loading model {model_name}...')

  torch.cuda.empty_cache()
  gpu_id = _get_largest_free_gpu()
  quantization_config = BitsAndBytesConfig(load_in_8bit=True)
  model_args = {
      'trust_remote_code': True,
      'torch_dtype': torch.float16,
      'device_map': f'cuda:{gpu_id}',
      'quantization_config': quantization_config,
  }

  match model_name:
    case 'deepseek-coder-7b-instruct-v1.5' | 'Qwen/Qwen2.5-Coder-7B-Instruct':
      tokenizer = AutoTokenizer.from_pretrained(config['models'][model_name], trust_remote_code=True)
      model = AutoModelForCausalLM.from_pretrained(config['models'][model_name], **model_args)
    case _:
      raise ValueError(f'{model_name} is unsupported yet.')

  if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

  return Translator(model_name, tokenizer, model)


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
  print(f'Translating from {src_lang} to {dst_lang}...')
  prompt = f'{config["prompts"]["prologue"]} \
             <source_language>{src_lang}</source_language> \
             <target_language>{dst_lang}</target_language> \
             <code>```{src_lang}\n%s```</code>'
  translated = [None] * len(snippets)

  gpu_id = _get_largest_free_gpu()
  for i, snippet in enumerate(snippets[:3]):
    messages = [{'role': 'user', 'content': prompt % snippet.code}]

    torch.cuda.empty_cache()

    match translator.name:
      case 'deepseek-coder-7b-instruct-v1.5' | 'Qwen/Qwen2.5-Coder-7B-Instruct':
        inputs = translator.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
        attention_mask = torch.ones_like(inputs).to(f'cuda:{gpu_id}')
        inputs = inputs.to(f'cuda:{gpu_id}')
      case _:
        raise ValueError(f'{translator.name} is unsupported yet.')

    outputs = translator.model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=512,
        do_sample=False,
        num_return_sequences=1,
        eos_token_id=translator.tokenizer.eos_token_id,
        pad_token_id=translator.tokenizer.pad_token_id,
        use_cache=True,
    )

    response = translator.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    matched = re.search(r'```\w+\n(.+)```', response, re.DOTALL)
    if not matched:
      raise ValueError(f'Incorrect format of {snippet.id}.')
    print(f'Snippet {i}:\n{matched.group(1)}\n')
    translated[i] = snippet._replace(code=matched.group(1))
  return translated
