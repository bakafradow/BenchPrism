from collections.abc import Sequence
from typing import NamedTuple


class Snippet(NamedTuple):
  id: str
  code: str
  args: dict = {}


TestBatch = Sequence[tuple[str, Sequence[str]]]
