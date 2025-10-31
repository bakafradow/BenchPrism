from abc import ABC, abstractmethod
from collections.abc import Sequence

from .. import Snippet


class BaseTransformer(ABC):
  """
  Abstract base class for code transformers.
  """

  @abstractmethod
  def transform(
    self,
    snippets: Sequence[Snippet],
    lang: str
  ) -> Sequence[Sequence[Snippet | None]]:
    """
    Transforms the given code snippet.
    :param snippets: the code snippets to transform
    :param lang: the language of the code snippet
    :return: the transformed code snippets
    """
    return ((snippets,),)


def transformer_factory() -> BaseTransformer:
  from .stylex import StyleX
  return StyleX()
