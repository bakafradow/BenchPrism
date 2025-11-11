import os
from argparse import ArgumentParser

from typing import Callable
import tiktoken
from google import genai
from tqdm import tqdm
from functools import cache

from scripts.ipython_setup import _load_benchmark
from stylo_flora import Snippet
from stylo_flora.inference.tasks import (code_repair, code_summarization,
                                         code_translation, io_reasoning,
                                         mcq_answering, tag_classification,
                                         test_generation)
from transformers import AutoTokenizer


def get_prompt_code_translation(snippet: Snippet, lang: str) -> tuple[str, str]:
  return code_translation.SYSTEM_PROMPT, code_translation.USER_PROMPT.format(
    src_lang=lang,
    dst_lang='python',
    code=snippet.code,
  )


def get_prompt_code_repair(snippet: Snippet, lang: str) -> tuple[str, str]:
  return code_repair.SYSTEM_PROMPT, code_repair.USER_PROMPT.format(
    lang=lang,
    desc=snippet.args['desc'],
    input_spec=snippet.args['input_spec'],
    output_spec=snippet.args['output_spec'],
    sample_inputs=repr(snippet.args['sample_inputs']),
    sample_outputs=repr(snippet.args['sample_outputs']),
    code=snippet.code,
    msg=snippet.args.get('error_msg', 'No error message provided.'),
  )


def get_prompt_code2tag(snippet: Snippet, lang: str) -> tuple[str, str]:
  return tag_classification.SYSTEM_PROMPT, tag_classification.USER_PROMPT.format(
    lang=lang,
    code=snippet.code,
  )


def get_prompt_descode2tag(snippet: Snippet, lang: str) -> tuple[str, str]:
  return tag_classification.SYSTEM_PROMPT, tag_classification.USER_PROMPT.format(
    lang=lang,
    code=snippet.code,
  ) + f'\n<problem_description>{snippet.args["desc"]}</problem_description>\n'


def get_prompt_code_summarization(snippet: Snippet, lang: str) -> tuple[str, str]:
  return code_summarization.SYSTEM_PROMPT, code_summarization.USER_PROMPT.format(
    lang=lang,
    code=snippet.code,
  )


def get_prompt_input_reasoning(snippet: Snippet, lang: str) -> tuple[str, str]:
  return io_reasoning.SYSTEM_PROMPT, io_reasoning.USER_PROMPT.format(
    lang=lang,
    code=snippet.code,
  )


def get_prompt_output_reasoning(snippet: Snippet, lang: str) -> tuple[str, str]:
  return io_reasoning.SYSTEM_PROMPT, io_reasoning.USER_PROMPT.format(
    lang=lang,
    code=snippet.code,
  )


def get_prompt_mcq_answering(snippet: Snippet, lang: str) -> tuple[str, str]:
  return mcq_answering.SYSTEM_PROMPT, mcq_answering.USER_PROMPT.format(
    lang=lang,
    code=snippet.code,
    choices='\n'.join([f'{letter}. {content}' for letter, content in zip('ABCD', snippet.args['choices'])]),
  )


def get_prompt_test_generation(snippet: Snippet, lang: str) -> tuple[str, str]:
  return test_generation.SYSTEM_PROMPT, test_generation.USER_PROMPT.format(
    lang=lang,
    desc=snippet.args['desc'],
    input_spec=snippet.args['input_spec'],
    output_spec=snippet.args['output_spec'],
    sample_inputs=repr(snippet.args['sample_inputs']),
    sample_outputs=repr(snippet.args['sample_outputs']),
    code=snippet.code,
    notes=snippet.args['notes'],
  )


@cache
def _get_encoding_gpt5mini() -> tiktoken.Encoding:
  return tiktoken.encoding_for_model('gpt-5-mini')


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
def _get_client_gemini() -> genai.Client:
  return genai.Client(api_key=os.getenv('API_KEY'))


def count_tokens_gemini25flash(sys_prompt: str, user_prompt: str) -> int:
  client_gemini = _get_client_gemini()
  res = client_gemini.models.count_tokens(
    model='gemini-2.5-flash',
    config=genai.types.CountTokensConfig(
      # system_instruction=sys_prompt
    ),
    contents=user_prompt)
  return res.total_tokens if res.total_tokens else 0


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


def count_tokens_local(model_name: str) -> Callable:
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  def wrapper(sys_prompt: str, user_prompt: str) -> int:
    return _count_tokens_local(tokenizer, sys_prompt, user_prompt)
  return wrapper


def main() -> None:
  parser = ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, required=True,
                      help='Specify one dataset to evaluate.')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Specify the model to use.')
  parser.add_argument('--src-lang', type=str, required=True,
                      help='Specify the source language.')
  args = parser.parse_args()

  print(f'Counting tokens for model {args.model} on dataset {args.dataset}...')

  if 'gpt' in args.model:
    count_func = count_tokens_gpt5mini
  elif 'gemini' in args.model:
    count_func = count_tokens_gemini25flash
  elif 'qwen2.5' in args.model:
    count_func = count_tokens_local('Qwen/Qwen2.5-7B-Instruct')
  elif 'phi4' in args.model:
    count_func = count_tokens_local('microsoft/Phi-4-mini-reasoning')
  elif 'codegeex4' in args.model:
    count_func = count_tokens_local('THUDM/codegeex4-all-9b')
  elif 'codellama' in args.model:
    count_func = count_tokens_local('codellama/CodeLlama-7b-Instruct-hf')
  else:
    raise NotImplementedError(f'Model {args.model} not supported yet.')

  print('task'.ljust(20) + 'total tokens'.ljust(15) + 'avg tokens/snippet')
  for task in [
    'code_translation', 'code_repair', 'code2tag', 'descode2tag',
    'code_summarization', 'input_reasoning', # 'output_reasoning',
    'mcq_answering', 'test_generation']:
    params = [args.dataset, task, args.src_lang]
    if task == 'code_translation':
      params.append('python')
    try:
      snippets = _load_benchmark(*params)
    except NotImplementedError:
      print(f'Skipping {task}.')
      continue
    total_tokens = 0
    for snippet in tqdm(snippets, desc=f'Counting on {task}', leave=False):
      get_prompt_func = globals()[f'get_prompt_{task}']
      sys_prompt, user_prompt = get_prompt_func(snippet, args.src_lang)
      total_tokens += count_func(sys_prompt, user_prompt)
    print(f'{task.ljust(20)}{str(total_tokens).ljust(15)}{total_tokens / len(snippets):.2f}')


if __name__ == '__main__':
  main()
