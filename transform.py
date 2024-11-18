import pandas as pd
from typing import List


def transform_exhaustively(source_code: pd.DataFrame) -> List[pd.DataFrame]:
  """
  Transform source code with Java-implemented transformer which is implemented in Java, applying variant
  :param source_code: array of source code
  :return: a set of code which indicates different combinations of mutations
  """
  # TODO:
  #  1. Figure out how to determine all the possible mutations given the source code
  #  2. Figure out how to apply specific mutation with the transformer
  #  3. New interface in transformer to pass the source code read in Python to the transformer implemented in Java
  #  4. Construct the transformed code set and return it
  pass
