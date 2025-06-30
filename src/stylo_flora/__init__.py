from collections.abc import Sequence
from typing import NamedTuple


class Snippet(NamedTuple):
  id: str
  code: str
  ref: str = ''


TestBatch = Sequence[tuple[str, Sequence[str]]]
