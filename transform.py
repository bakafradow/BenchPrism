import jpype as jp
import pandas as pd
import yaml


def apply_one(snippets, transformer: jp.JClass) -> pd.DataFrame:
  df = pd.DataFrame(columns=['id'])
  df['id'] = snippets['id']
  df['snippet'] = pd.Series.apply(
    snippets['snippet'],
    lambda snippet: str(transformer.applyOne(snippet)))
  return df


def apply_all(snippets, transformer: jp.JClass) -> pd.DataFrame:
  df = pd.DataFrame(columns=['id'])
  df['id'] = snippets['id']
  df['snippet'] = pd.Series.apply(
    snippets['snippet'],
    lambda snippet: [str(x) for x in transformer.applyAll(snippet)])
  df.explode('snippet')
  return df


def transform_source(snippets: pd.DataFrame) -> pd.DataFrame:
  """
  Applies transformations to the source code to generate a set of mutated code with a code style transformer.
  :param snippets: the snippets to be transformed
  :return: a set of code which indicates different combinations of mutations
  """
  print('Applying transformations...')
  with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['transformer']
  jp.startJVM('-ea', jvmpath=config['jvmpath'],
              classpath=[config['classpath']])
  transformer_class = jp.JClass('org.example.experiment.StringTransformer')
  transformer = transformer_class()
  try:
    return apply_one(snippets, transformer)
  finally:
    jp.shutdownJVM()
