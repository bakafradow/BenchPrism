from typing import NamedTuple


class Snippet(NamedTuple):
  id: int
  code: str
  ref: str = ''
