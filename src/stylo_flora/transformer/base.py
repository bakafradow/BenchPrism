from abc import ABC, abstractmethod

from .. import Snippet


class BaseTransformer(ABC):
  """
  Abstract base class for code transformers.
  """

  @abstractmethod
  def transform(
      self,
      snippet: Snippet,
      *,
      check: bool = False,
      seqs_to_skip: set[int] = set(),
  ) -> list[str | None]:
    """
    Transforms coding styles of the given code snippet.

    :param snippet: the code snippet to transform
    :param check: whether to reject incorrectly transformed snippets
    :param seqs_to_skip: set of sequence indices to skip
    :return: a series of transformed code snippets
    """
    raise NotImplementedError


def transformer_factory(lang: str, seed: int) -> BaseTransformer:
  from .stylex import StyleX
  return StyleX(lang=lang, seed=seed)
