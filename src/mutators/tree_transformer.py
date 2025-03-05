import tree_sitter
import yaml

from . import CallbackType
from .callbacks import statement
from .mutator import BaseMutator
from .. import SnippetSequence


class TreeTransformer(BaseMutator):
  def __init__(self, src_lang):
    lang_to_name = {
      'java': 'java',
      'cpp': 'cpp',
      'cs': 'c_sharp',
      'python': 'python',
    }
    self.lang_name = lang_to_name[src_lang]
    self.language = tree_sitter.Language(__import__(f'tree_sitter_{self.lang_name}').language())
    self.parser = tree_sitter.Parser(self.language)

    with open('config/rules.yaml', 'r') as f:
      self.rules = yaml.safe_load(f)['rules']

  def apply_one(self, snippets: SnippetSequence) -> SnippetSequence:
    with open('config/settings.yaml', 'r') as f:
      config = yaml.safe_load(f)['mutator']
    return [snippet._replace(code=self._apply_rule(snippet.code, config['rule_id'])) for snippet in snippets]

  def _apply_rule(self, code: str, id: int) -> str:
    tree = self.parser.parse(bytes(code, 'utf-8'))
    # TODO: another layer that convert language-agnostic patterns into language-specific ones
    match self.rules[id]['granularity']:
      case 'statement':
        callback: CallbackType = getattr(statement, self.rules[id]['callback'])
        for src, dst in self.rules[id]['edges']:
          query = self.language.query(self.rules[id]['patterns'][src])
          while len(matches := query.matches(tree.root_node)) != 0:
            match = matches[0]
            stmt_node = match[1]['stmt'][0]
            mutant = callback(code, match, dst)
            code = code[:stmt_node.start_byte] + mutant + code[stmt_node.end_byte:]
            new_end_byte = stmt_node.start_byte + len(mutant)
            new_end_row = stmt_node.start_point.row + mutant.count('\n')
            new_end_col = len(mutant.split('\n')[-1]) if '\n' in mutant else stmt_node.start_point.column + len(mutant)
            tree.edit(
              start_byte=stmt_node.start_byte,
              old_end_byte=stmt_node.end_byte,
              new_end_byte=new_end_byte,
              start_point=stmt_node.start_point,
              old_end_point=stmt_node.end_point,
              new_end_point=(new_end_row, new_end_col)
            )
            tree = self.parser.parse(bytes(code, 'utf-8'), tree)
        # TODO: verify semantic perseverance
      case _:
        raise ValueError('Unimplemented granularity.')
    return code
