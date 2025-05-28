from abc import ABC, abstractmethod
from collections.abc import Sequence

from .. import Snippet


class BaseTransformer(ABC):
  """
  Abstract base class for code transformers.
  """

  @abstractmethod
  def transform(self, snippets: Sequence[Snippet], lang: str) -> Sequence[Sequence[Snippet | None]]:
    """
    Transforms the given code snippet.
    :param snippet: the code snippet to transform
    :return: the transformed code snippet
    """
    return tuple((snippets,))


def transformer_factory() -> BaseTransformer:
  from .egsi import EGSI
  return EGSI()
