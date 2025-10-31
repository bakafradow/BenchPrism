"""
Attempts to span a snippet using the StyleX transformer.
"""

import os
import sys

import jpype as jp
from dotenv import load_dotenv

from stylo_flora.transformer.stylex import StyleX

load_dotenv()

stylex = StyleX()


def span(snippet: str, seq: list[int]) -> str:
  seq_list = jp.java.util.List.of(*[jp.java.lang.Integer(num) for num in seq])
  mutant = stylex.cls.span("java", snippet, seq_list)
  seq_list = None
  jp.java.lang.System.gc()
  return str(mutant)


def main():
  if len(sys.argv) < 3:
    print(f'Usage: python {os.path.basename(__file__)} <input file> <output file> [style file]')
    sys.exit(1)
  with open(sys.argv[1], 'r', encoding='utf-8') as f:
    snippet = f.read()
    print(f'Read snippet with {len(snippet.splitlines())} lines from {sys.argv[1]}')
  seq = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 1, 0, -1, -1]

  mutant_str = span(snippet, seq)

  with open(sys.argv[2], 'w', encoding='utf-8') as f:
    f.write(mutant_str)
    print(f'Wrote mutant with {len(mutant_str.splitlines())} lines to {sys.argv[2]}')


if __name__ == '__main__':
  main()
