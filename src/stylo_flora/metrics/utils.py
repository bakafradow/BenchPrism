import re

from .. import Snippet


def extract_classname_java(snippet: Snippet) -> str | None:
  matched = re.search(r'public\s+(?:final\s+)?class\s+(\w+)', snippet.code)
  if not matched:
    matched = re.search(r'(?:final\s+)?class\s+(\w+)', snippet.code)
  if not matched:
    return None
  return matched.group(1)
