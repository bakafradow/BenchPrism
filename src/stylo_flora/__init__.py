from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import NamedTuple


@dataclass
class Snippet:
  """
  The unit that forms the workflow.

  @param id: The identifier of the snippet which is unique among a dataset.
  @param code: The content of the snippet.
  @param args: Additional arguments for the snippet. Note that the content of `args` will **NOT** be deepcopied.
  """
  id: str
  code: str
  args: dict = field(default_factory=dict)

  def __deepcopy__(self, memo: dict) -> 'Snippet':
    new_args = self.args.copy()
    snippet = Snippet(id=self.id, code=self.code, args=new_args)
    memo[id(self)] = snippet
    return snippet

  def replace(self, **kwargs) -> 'Snippet':
    snippet_dict = asdict(deepcopy(self))
    snippet_dict.update(kwargs)
    return Snippet(**snippet_dict)


class IOTestCase(NamedTuple):
  input: str
  outputs: Sequence[str]


class APITestCase(NamedTuple):
  """If `method` is empty, all methods in the file will be tested."""
  file: str
  code: str
  method: str = ''
