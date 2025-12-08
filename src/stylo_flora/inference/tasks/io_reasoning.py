"""
Modified from CRUXEval-X repository (https://github.com/CRUXEVAL-X/cruxeval-x).
"""

import re
import sys
from enum import Enum, auto
from functools import cache
from typing import Any

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from ... import Snippet
from ...logger import logger
from .base import BaseTask


class ReasoningType(str, Enum):
  @staticmethod
  def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> Any:
    return name.lower()

  INPUT_REASONING = auto()
  OUTPUT_REASONING = auto()


MASK = '????'


@cache
def _get_parser(lang: str) -> Parser:
  logger.info(f'Initializing Tree-sitter {lang} parser...')
  if lang == 'java':
    return Parser(Language(tsjava.language()))
  else:
    raise ValueError(f'Unsupported language: {lang}')


def mask_java(code: str, type: ReasoningType) -> str:
  parser = _get_parser('java')
  code_bytes = bytes(code, encoding='utf8')
  tree = parser.parse(code_bytes)
  try:
    class_node = next(node for node in tree.root_node.children if node.type == 'class_declaration')
    class_body_node = next(node for node in class_node.children if node.type == 'class_body')
    main_method_node = next(node for node in class_body_node.children
                            if node.type == 'method_declaration' and node.child(2).text == b'main')
    main_block_node = next(node for node in main_method_node.children if node.type == 'block')
    assert_node = next(node for node in main_block_node.children if node.type == 'assert_statement')
  except StopIteration as e:
    logger.error(f'Failed to find assertion statement in the main method: {e}')
  mask = bytes(MASK, encoding='utf8')
  match type:
    case ReasoningType.INPUT_REASONING:
      arg_list_node = assert_node.child(1).child(1).child(0).child(1)
      result_bytes = code_bytes[:arg_list_node.child(1).start_byte] + mask + \
          code_bytes[arg_list_node.child(arg_list_node.child_count - 1).start_byte:]
    case ReasoningType.OUTPUT_REASONING:
      if assert_node.child(1).child(1).type == 'method_invocation':
        arg_list_node = assert_node.child(1).child(1).child(3)
        result_bytes = code_bytes[:arg_list_node.child(1).start_byte] + mask + \
            code_bytes[arg_list_node.child(arg_list_node.child_count - 1).start_byte:]
      elif assert_node.child(1).child(1).type == 'binary_expression':
        binary_exp_node = assert_node.child(1).child(1)
        result_bytes = code_bytes[:binary_exp_node.child(2).start_byte] + mask + \
            code_bytes[binary_exp_node.child(binary_exp_node.child_count - 1).end_byte:]
      else:
        raise ValueError(f'Unexpected pattern in assertion statement: f{assert_node.text.decode()}')
    case _:
      raise ValueError(f'Unsupported type: {type}')
  return result_bytes.decode(encoding='utf8')


def mask(code: str, lang: str, type_: ReasoningType) -> str:
  mask_func = getattr(sys.modules[__name__], f'mask_{lang}', None)
  if not mask_func:
    raise ValueError(f'Unsupported language: {lang}')
  return mask_func(code, type_)


class IOReasoning(BaseTask):
  SYSTEM_PROMPT = f"""<task>
Based on the given code, which may contain errors, reason out the "{MASK}" in the assertion statement to make it compilable and correct in Java 17.
</task>
<constraint>
Your output MUST only contain the exact expression that should replace the "{MASK}" in the assertion statement instead of the whole assertion statement, enclosed by triple back quotes with the language specified without any explanation.
</constraint>
"""

  USER_PROMPT = """<code>```{lang}
{code}
```</code>
"""

  def __init__(self, lang: str, type_: ReasoningType) -> None:
    super().__init__()
    self.lang = lang
    self.type = type_

  def get_prompt(self, snippet: Snippet) -> tuple[str, str] | None:
    masked_code = mask(snippet.data['code'], self.lang, self.type)
    if masked_code.count(MASK) != 1:
      logger.warning(f'Failed to find exactly one "{MASK}" in masked code.')
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        lang=self.lang,
        code=masked_code,
    )

  def resolve_response(self, res: str) -> str | None:
    matched = re.search(r'```(?:\w+)?\n(.+)```', res, re.DOTALL)
    if not matched:
      logger.debug(res)
      return None
    return matched.group(1).strip()
