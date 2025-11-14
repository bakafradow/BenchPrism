from collections.abc import Sequence as Seq
from dataclasses import dataclass, field
from typing import NamedTuple

import yaml
from dotenv import load_dotenv

load_dotenv()

with open('configs/settings.yaml', 'r') as f:
  setting_dict = yaml.safe_load(f)


@dataclass
class Snippet:
  """
  The unit that forms the workflow.

  @param id: The identifier of the snippet which is unique among a dataset.
  @param code: The content of the snippet.
  @param args: Additional arguments for the snippet. Note that the content of `args` will **NOT** be deepcopied.
  """
  id: str
  data: dict = field(default_factory=dict)

  def replace(self, **kwargs) -> 'Snippet':
    data_dict = self.data.copy()
    data_dict.update(kwargs)
    return Snippet(id=self.id, data=data_dict)


class IOTestCase(NamedTuple):
  input: str
  outputs: Seq[str]


class APITestCase(NamedTuple):
  """If `method` is empty, all methods in the file will be tested."""
  file: str
  code: str
  method: str = ''
