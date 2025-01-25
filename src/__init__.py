from typing import NamedTuple, TypeAlias, Sequence


class Snippet(NamedTuple):
  id: int
  code: str


SnippetSequence: TypeAlias = Sequence[Snippet]
