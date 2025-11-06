from abc import ABC, abstractmethod
from collections.abc import MutableSequence as MSeq
from collections.abc import Sequence as Seq

from .. import Snippet


class BaseTransformer(ABC):
  """
  Abstract base class for code transformers.
  """

  @abstractmethod
  def transform(
    self,
    lang: str,
    snippets: Seq[Snippet],
    **kwargs,
  ) -> MSeq[MSeq[Snippet | None]]:
    """
    Transforms the given code snippet.
    :param snippets: the code snippets to transform
    :param lang: the language of the code snippet
    :return: the transformed code snippets
    """
    return [list(snippets)]


def transformer_factory() -> BaseTransformer:
  from .stylex import StyleX
  return StyleX()
