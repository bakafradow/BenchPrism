import ast
import re
from collections.abc import Mapping

import jpype as jp
from dotenv import load_dotenv

from stylo_flora.transformer.stylex import StyleX

load_dotenv()


stylex = StyleX(lang='java')
GlobalInfo = jp.JClass('org.example.global.GlobalInfo')
Configuration = jp.JClass('org.example.Configuration')


def span_single(lang: str, code: str, choice_dict: Mapping) -> str | None:
  GlobalInfo.setConf(Configuration())
  GlobalInfo.setLanguage(lang)
  styler_container = stylex._build_styler_container(lang, choice_dict)
  return stylex._apply_styles(lang, code, styler_container)


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
