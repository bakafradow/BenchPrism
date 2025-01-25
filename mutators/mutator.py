from snippet import SnippetSequence


class BaseMutator:
  @classmethod
  def apply_one(cls, snippets: SnippetSequence) -> SnippetSequence: ...
