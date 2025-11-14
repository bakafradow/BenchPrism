"""
Modified from CRUXEval-X repository (https://github.com/CRUXEVAL-X/cruxeval-x).
"""

import re
from enum import Enum, auto

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from ... import Snippet
from ...logger import logger
from .base import BaseTask


class ReasoningType(Enum):
  INPUT = auto()
  OUTPUT = auto()


class IOReasoning(BaseTask):
  SYSTEM_PROMPT = """
  <task>
  Based on the given code, which may contain errors, reason out the "????" in the assertion statement to make it compilable and correct in Java 17.
  </task>
  <constraint>
  Your output MUST only contain the exact expression that should replace the "????" in the assertion statement without any explanations, enclosed by triple back quotes with the language specified.
  </constraint>
  """

  USER_PROMPT = """
  <code>```{lang}
  {code}
  ```</code>
  """

  def __init__(self, lang: str, type: ReasoningType) -> None:
    super().__init__()
    self.lang = lang
    self.type = type

    logger.info('Initializing Tree-sitter parser...')
    match lang:
      case 'java':
        self.parser = Parser(Language(tsjava.language()))
        self.mask_func = self._mask_java
      case _:
        raise ValueError(f'Unsupported language: {lang}')

  def get_prompt(self, snippet: Snippet) -> tuple[str, str]:
    masked_code = self.mask_func(snippet.data['code'])
    return self.SYSTEM_PROMPT, self.USER_PROMPT.format(
        lang=self.lang,
        code=masked_code,
    )

  def resolve_response(self, res: str) -> str | None:
    matched = re.search(r'```(?:\w+)?\n(.+)```', res, re.DOTALL)
    if not matched:
      logger.debug(res)
      return None
    return matched.group(1)

  def _mask_java(self, code: str) -> str:
    code_bytes = bytes(code, encoding='utf8')
    tree = self.parser.parse(code_bytes)
    try:
      class_node = next(node for node in tree.root_node.children if node.type == 'class_declaration')
      class_body_node = next(node for node in class_node.children if node.type == 'class_body')
      main_method_node = next(node for node in class_body_node.children
                              if node.type == 'method_declaration' and node.child(2).text == b'main')
      main_block_node = next(node for node in main_method_node.children if node.type == 'block')
      assert_node = next(node for node in main_block_node.children if node.type == 'assert_statement')
    except StopIteration as e:
      logger.error(f'Failed to find assertion statement in the main method: {e}')
    mask = b'????'
    match self.type:
      case ReasoningType.INPUT:
        arg_list_node = assert_node.child(1).child(1).child(0).child(1)
        result_bytes = code_bytes[:arg_list_node.child(1).start_byte] + mask + \
            code_bytes[arg_list_node.child(arg_list_node.child_count - 1).start_byte:]
      case ReasoningType.OUTPUT:
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
