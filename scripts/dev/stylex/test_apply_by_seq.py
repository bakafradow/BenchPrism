"""
Attempts to span a snippet using the StyleX transformer with a sequence of choices which is written in the first line of the input file as a comment.
"""

import os
import sys

from scripts.dev.stylex.utils import find_seq, span_single, stylex


def main():
  if len(sys.argv) < 3:
    print(f'Usage: python {os.path.basename(__file__)} <lang> <input file> [output file]')
    sys.exit(1)
  lang = sys.argv[1]
  input_file = sys.argv[2]
  with open(input_file, 'r', encoding='utf-8') as f:
    code = f.read()
    print(f'Read {lang} code with {len(code.splitlines())} lines from {input_file}')

  seq = find_seq(lang, code)
  choice_dict = stylex._create_choice_dict(seq)
  variant_code = span_single(lang, code, choice_dict)

  if len(sys.argv) < 3:
    print(f"""
=========== VARIANT ============
{variant_code}
================================""")
    return
  output_file = sys.argv[3]
  with open(output_file, 'w', encoding='utf-8') as f:
    f.write(variant_code)
    print(f'Wrote variant with {len(variant_code.splitlines())} lines to {output_file}')


if __name__ == '__main__':
  main()
