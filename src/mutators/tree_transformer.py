import tree_sitter
import yaml

from .mutator import BaseMutator
from ..snippet import SnippetSequence


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

  def apply_one(self, snippets: SnippetSequence) -> SnippetSequence:
    with open('config/settings.yaml', 'r') as f:
      config = yaml.safe_load(f)['mutator']
    return [snippet._replace(code=self._apply_rule(snippet.code, config['rule'])) for snippet in snippets]

  def _apply_rule(self, code: str, rule: int) -> str:
    # builds AST from code
    tree = self.parser.parse(bytes(code, 'utf-8'))
    # converts patterns of a language-independent rule to queries
    # finds occurrences in AST with the queries
    # transforms all occurrences according to one pattern of the rule
    return code
