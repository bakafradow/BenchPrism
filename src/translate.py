from typing import Sequence

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import Snippet

with open('config/settings.yaml') as f:
  config = yaml.safe_load(f)['translator']


def translate_with_model(snippets: Sequence[Snippet], model: str, src_lang: str, dst_lang: str) -> Sequence[Snippet]:
  """
  Translates snippets in code set with code translation model.
  :param snippets: the snippets to be translated
  :param model: name of the code translation model
  :param src_lang: source language
  :param dst_lang: destination language
  :return: a set of translated code
  """
  print(f'Translating from {src_lang} to {dst_lang} with {model}...')
  prompt = f'Translate the following {src_lang} code to {dst_lang}. \
             Output only the translated code, no explanations. \
             Comments are allowed.\n\n{src_lang} code:\n```{src_lang.lower()}%s```'
  translated = [None] * len(snippets)
  match model:
    case 'deepseek-coder-7b-instruct-v1.5':
      tokenizer = AutoTokenizer.from_pretrained(f'{config['models'][model]}', trust_remote_code=True)
      model = AutoModelForCausalLM.from_pretrained(f'{config['models'][model]}', trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
      for i, snippet in enumerate(snippets):
        messages = [
          { 'role': 'user',
            'content': prompt % snippet.code
          }
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        translated[i] = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(translated[i])
    case _:
      raise ValueError(f'{model} is unsupported yet.')
  return translated
