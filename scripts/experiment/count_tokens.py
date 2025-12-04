import os
from argparse import ArgumentParser
from functools import cache
from typing import Callable

import tiktoken
from google import genai
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.experiment.utils import load_snippets
from stylo_flora.inference import task_factory


def count_tokens_gpt5mini(sys_prompt: str, user_prompt: str) -> int:
  encoding_gpt5mini = _get_encoding_gpt5mini()
  tokens_per_msg = 3
  tokens_per_name = 1
  num_tokens = 0
  msgs = [
      {'role': 'system', 'content': sys_prompt},
      {'role': 'user', 'content': user_prompt}
  ]
  for msg in msgs:
    num_tokens += tokens_per_msg
    for k, v in msg.items():
      num_tokens += len(encoding_gpt5mini.encode(v))
      if k == 'name':
        num_tokens += tokens_per_name
  return num_tokens


@cache
def _get_encoding_gpt5mini() -> tiktoken.Encoding:
  return tiktoken.encoding_for_model('gpt-5-mini')


def count_tokens_gemini25flash(sys_prompt: str, user_prompt: str) -> int:
  client_gemini = _get_client_gemini()
  res = client_gemini.models.count_tokens(
      model='gemini-2.5-flash',
      config=genai.types.CountTokensConfig(
          # system_instruction=sys_prompt
      ),
      contents=user_prompt)
  return res.total_tokens if res.total_tokens else 0


@cache
def _get_client_gemini() -> genai.Client:
  return genai.Client(api_key=os.getenv('API_KEY'))


def count_tokens_local(model_name: str) -> Callable:
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

  def wrapper(sys_prompt: str, user_prompt: str) -> int:
    return _count_tokens_local(tokenizer, sys_prompt, user_prompt)
  return wrapper


def _count_tokens_local(tokenizer, sys_prompt: str, user_prompt: str) -> int:
  input_ids = tokenizer.apply_chat_template(
      conversation=[
          {'role': 'system', 'content': sys_prompt},
          {'role': 'user', 'content': user_prompt}
      ],
      tokenize=True,
      add_generation_prompt=True,
      return_tensors='pt',
  )
  return len(input_ids[0])


def main() -> None:
  print(f'Counting tokens for model {args.model} on dataset {args.dataset}...')

  if 'gpt' in args.model:
    count_func = count_tokens_gpt5mini
  elif 'gemini' in args.model:
    count_func = count_tokens_gemini25flash
  elif 'qwen2.5' in args.model:
    count_func = count_tokens_local('Qwen/Qwen2.5-7B-Instruct')
  elif 'phi-4' in args.model:
    count_func = count_tokens_local('microsoft/Phi-4-mini-reasoning')
  elif 'codegeex4' in args.model:
    count_func = count_tokens_local('THUDM/codegeex4-all-9b')
  elif 'codellama' in args.model:
    count_func = count_tokens_local('codellama/CodeLlama-7b-Instruct-hf')
  else:
    raise NotImplementedError(f'Model {args.model} not supported yet.')

  print('task'.ljust(20) + 'total tokens'.ljust(15) + 'avg tokens/snippet')
  for task_name in [
      'code_translation', 'code_repair', 'code2tag', 'descode2tag',
      'code_summarization', 'input_reasoning',
      'mcq_answering', 'test_generation',
  ]:
    args.task = task_name
    try:
      snippets = load_snippets(args)
    except NotImplementedError:
      print(f'Skipping {task_name}.')
      continue
    task = task_factory(task_name, **dict(args._get_kwargs()))
    total_tokens = 0
    for snippet in tqdm(snippets, desc=f'Counting on {task_name}', leave=False):
      sys_prompt, user_prompt = task.get_prompt(snippet)
      total_tokens += count_func(sys_prompt, user_prompt)
    print(f'{task_name.ljust(20)}{str(total_tokens).ljust(15)}{total_tokens / len(snippets):.2f}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model to use.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  args = parser.parse_args()
  args.dst_lang = 'python'

  main()
