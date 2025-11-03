"""
Attempts to span a snippet using the StyleX transformer.
"""

import json
import os
import sys

import jpype as jp
from dotenv import load_dotenv

load_dotenv()

from stylo_flora.transformer.stylex import StyleX
stylex = StyleX()
GlobalInfo = jp.JClass('org.example.global.GlobalInfo')
Configuration = jp.JClass('org.example.Configuration')


def span_single(lang: str, code: str, choice_dict: dict) -> str:
  GlobalInfo.setConf(Configuration())
  GlobalInfo.setLanguage(lang)
  return stylex._apply_styles_by_choices(lang, code, choice_dict)


def main():
  if len(sys.argv) < 4:
    print(f'Usage: python {os.path.basename(__file__)} <lang> <input file> <choice file> [output file]')
    sys.exit(1)
  lang = sys.argv[1]
  input_file = sys.argv[2]
  choice_file = sys.argv[3]
  with open(input_file, 'r', encoding='utf-8') as f:
    code = f.read()
    print(f'Read {lang} code with {len(code.splitlines())} lines from {input_file}')
  with open(choice_file, 'r', encoding='utf-8') as f:
    choice_dict = json.load(f)
    print(f'Read choice dict with {len(choice_dict)} entries from {choice_file}')

  variant_code = span_single(lang, code, choice_dict)

  if len(sys.argv) < 4:
    print(f"""
=========== VARIANT ============
{variant_code}
================================""")
    return
  output_file = sys.argv[4]
  with open(output_file, 'w', encoding='utf-8') as f:
    f.write(variant_code)
    print(f'Wrote variant with {len(variant_code.splitlines())} lines to {output_file}')


if __name__ == '__main__':
  main()
