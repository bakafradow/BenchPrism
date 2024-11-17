import json
import numpy as np
from attr import dataclass
from datasets import load_dataset
from pprint import pprint


def extract(dataset: str):
  """
  Extracts source code from different datasets into unified format.
  :param dataset: dataset name
  :return: source code extracted from the dataset
  """
  print(f'Extracting source code from {dataset} dataset...')
  match dataset:
    case 'HumanEvalX': # languages: python, cpp, go, java, js
      ds = load_dataset('THUDM/humaneval-x', 'java', trust_remote_code=True)
      # TODO:
      #  1. concatenate the 'declaration' and 'canonical_solution' fields into a single 'source_code' field
      #  2. extract the 'task_id' and 'source_code' fields
      return ds
    case 'xCodeEval': # languages: c, cpp, cs, go, java, js, kotlin, php, python, ruby, rust
      # TODO:
      #  1. select rows where the 'lang_cluster' field is 'Java'
      #  2. extract the 'src_uid' and 'source_code' fields
      ds = load_dataset('NTU-NLP-sg/xCodeEval', 'code_translation', trust_remote_code=True, streaming=True)
      # get all possible values of 'lang_cluster'
      return ds
    case 'XLCoST': # languages: c, cs, cpp, java, js, php, python
      # TODO:
      #  1. extract the 'source' and 'target' fields
      raise NotImplementedError('XLCoST is not supported yet')
    case 'CodeXGLUE': # languages: cs, java
      # TODO:
      #  1. extract the 'id' and 'java' fields
      ds = load_dataset('google/code_x_glue_cc_code_to_code_trans', trust_remote_code=True)
      return ds
    case 'G-TransEval': # languages: cpp, java, python
      # TODO:
      #  1. extract the 'id' and 'content' fields
      ds = load_dataset(f'xin1997/g-transeval-{lang}_all_only_input', trust_remote_code=True)
      return ds
    case _:
      raise ValueError(f'Unknown dataset: {dataset}.')
