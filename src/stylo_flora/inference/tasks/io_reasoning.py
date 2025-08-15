"""
Modified from CRUXEval-X repository (https://github.com/CRUXEVAL-X/cruxeval-x).
"""

import re
from collections.abc import Sequence

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser

from ... import Snippet
from ...logger import logger
from ..agents import BaseAgent, Prompt
from ..utils import work

logger.info('Initializing Tree-sitter parser...')
_JAVA_LANGUAGE = Language(tsjava.language())
_PARSER = Parser(_JAVA_LANGUAGE)

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


def _reason(agent: BaseAgent, snippets: Sequence[Snippet], lang: str) -> Sequence[Snippet | None]:
  def worker(i: int, snippet: Snippet) -> Snippet | None:
    prompt = Prompt(id=snippet.id, system=SYSTEM_PROMPT,
                    user=USER_PROMPT.format(lang=lang, code=snippet.code))
    response = agent.generate(prompt)
    matched = re.search(r'```(?:\w+)?\n(.+)```', response, re.DOTALL)
    if not matched:
      logger.warning(f'Content of {snippet.id} not found.')
      logger.debug(response)
      return None
    logger.debug(f'Content for snippet {i}: {response}')
    return snippet.replace(code=snippet.code.replace('????', matched.group(1)))

  return work(worker=worker, snippets=snippets)


def _mask_java(code: str, type: str) -> str:
  code_bytes = bytes(code, encoding='utf8')
  tree = _PARSER.parse(code_bytes)
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
  match type:
    case 'input':
      arg_list_node = assert_node.child(1).child(1).child(0).child(1)
      result_bytes = code_bytes[:arg_list_node.child(1).start_byte] + mask + \
          code_bytes[arg_list_node.child(arg_list_node.child_count - 1).start_byte:]
    case 'output':
      if assert_node.child(1).child(1).type == 'method_invocation':
        arg_list_node = assert_node.child(1).child(1).child(3)
        result_bytes = code_bytes[:arg_list_node.child(1).start_byte] + mask + \
            code_bytes[arg_list_node.child(arg_list_node.child_count - 1).start_byte:]
      elif assert_node.child(1).child(1).type == 'binary_expression':
        binary_exp_node = assert_node.child(1).child(1)
        result_bytes = code_bytes[:binary_exp_node.child(2).start_byte] + mask + \
            code_bytes[binary_exp_node.child(binary_exp_node.child_count - 1).end_byte:]
      else:
        raise ValueError(f'Unexpected pattern in assertion statement: f{assert_node.text}')
    case _:
      raise TypeError(f'Unsupported type: {type}')
  return result_bytes.decode(encoding='utf8')


def _mask(lang: str, code: str, type: str) -> str:
  """
  Masks input/output part of the assertion statement in the main method with "????", adapted for CRUXEval-X.
  Note that in CRUXEval-X, each main method contains exactly one assertion statement.
  For example, `assert(f((6173l)).equals(("Not found")))` will become `assert(????).equals(("Not found"))`.

  @param type: `input` or `output`
  """
  mask_func = globals().get(f'_mask_{lang}')
  if not mask_func:
    raise ValueError(f'Unsupported language: {lang}')
  return mask_func(code, type)


def reason_input(agent: BaseAgent, snippets: Sequence[Snippet | None], lang: str) -> Sequence[Snippet | None]:
  snippets_without_input = [snippet.replace(code=_mask(lang, snippet.code, type='input'))
                            if snippet else None for snippet in snippets]
  return _reason(agent, snippets_without_input, lang)


def reason_output(agent: BaseAgent, snippets: Sequence[Snippet | None], lang: str) -> Sequence[Snippet | None]:
  snippets_without_output = [snippet.replace(code=_mask(lang, snippet.code, type='output'))
                             if snippet else None for snippet in snippets]
  return _reason(agent, snippets_without_output, lang)
