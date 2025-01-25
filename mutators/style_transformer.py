import jpype as jp
import yaml

from snippet import SnippetSequence
from mutators.mutator import BaseMutator


class CodeStyleTransformer(BaseMutator):
  def __init__(self, src_lang):
    with open('config.yaml', 'r') as f:
      config = yaml.safe_load(f)['mutator']
    jp.startJVM('-ea', jvmpath=config['jvmpath'],
                classpath=[config['classpath']])
    transformer_class = jp.JClass(config['class'])
    self.transformer = transformer_class(src_lang)

  def __del__(self):
    jp.shutdownJVM()

  def apply_one(self, snippets: SnippetSequence) -> SnippetSequence:
    return [snippet._replace(code=str(self.transformer.applyOne(snippet.code))) for snippet in snippets]
