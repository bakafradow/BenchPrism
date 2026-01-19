from abc import ABC, abstractmethod
from typing import Any

from ... import Snippet


class BaseTask(ABC):
  """
  Abstract base class for inference tasks.
  """

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if not hasattr(cls, 'SYSTEM_PROMPT'):
      raise NotImplementedError(f'{cls.__name__} must define \'SYSTEM_PROMPT\'.')
    if not hasattr(cls, 'USER_PROMPT'):
      raise NotImplementedError(f'{cls.__name__} must define \'USER_PROMPT\'.')

  @abstractmethod
  def get_prompt(self, snippet: Snippet) -> tuple[str, str] | None:
    """
    Obtains the system and user prompts for the given snippet.

    :param snippet: the code snippet with necessary information
    :return: a tuple consisting of a system prompt and a user prompt
    """
    raise NotImplementedError

  @abstractmethod
  def resolve_response(self, res: str) -> Any:
    """
    Resolves the response from the model.

    :param res: the raw response from the model
    :return: the processed response
    """
    raise NotImplementedError
