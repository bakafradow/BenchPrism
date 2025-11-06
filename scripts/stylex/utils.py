from stylo_flora.transformer.stylex import StyleX
import ast
import re

import jpype as jp
from dotenv import load_dotenv

load_dotenv()


stylex = StyleX()
GlobalInfo = jp.JClass('org.example.global.GlobalInfo')
Configuration = jp.JClass('org.example.Configuration')


def span_single(lang: str, code: str, choice_dict: dict) -> str:
  GlobalInfo.setConf(Configuration())
  GlobalInfo.setLanguage(lang)
  return stylex._apply_styles_by_choices(lang, code, choice_dict)


def find_seq(lang: str, code: str) -> list[int] | None:
  match lang:
    case 'cpp' | 'java':
      prefix = r'//'
    case 'python':
      prefix = r'#'
    case _:
      raise ValueError(f'Unsupported language: {lang}')
  matched = re.search(prefix + r'\s+Seq\s*=\s*(\[[^\]]*\])', code)
  if not matched:
    return None
  seq_str = matched.group(1)
  return ast.literal_eval(seq_str)
