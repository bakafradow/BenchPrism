import re


def extract_classname_java(code: str) -> str | None:
  matched = re.search(r'public\s+(?:final\s+)?class\s+(\w+)', code)
  if not matched:
    matched = re.search(r'(?:final\s+)?class\s+(\w+)', code)
  if not matched:
    return None
  return matched.group(1)
