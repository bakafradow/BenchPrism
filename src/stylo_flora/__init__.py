from collections.abc import Sequence
from typing import NamedTuple


class Snippet(NamedTuple):
  """
  The unit that forms the workflow.

  @param id: The identifier of the snippet which is unique among a dataset.
  @param code: The content of the snippet.
  @param args: Additional arguments for the snippet. Note that the content of `args` will **NOT** be deepcopied.
  """
  id: str
  code: str
  args: dict = {}


class IOTestCase(NamedTuple):
  input: str
  outputs: Sequence[str]


class APITestCase(NamedTuple):
  """If `method` is empty, all methods in the file will be tested."""
  file: str
  code: str
  method: str = ''
