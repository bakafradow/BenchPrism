from abc import ABC, abstractmethod
from collections.abc import Sequence

from .. import Snippet, TestBatch


class BaseTransformer(ABC):
  """
  Abstract base class for code transformers.
  """

  @abstractmethod
  def transform(
    self,
    snippets: Sequence[Snippet],
    test_batches: Sequence[TestBatch],
    lang: str
  ) -> Sequence[Sequence[Snippet | None]]:
    """
    Transforms the given code snippet.
    :param snippet: the code snippet to transform
    :param test_batches: the test cases for the code snippet
    :param lang: the language of the code snippet
    :return: the transformed code snippet
    """
    return tuple((snippets,))


def transformer_factory() -> BaseTransformer:
  from .egsi import EGSI
  return EGSI()
