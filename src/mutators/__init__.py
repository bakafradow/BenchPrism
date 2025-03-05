from typing import TypeAlias, Mapping, Callable

import tree_sitter

MatchType: TypeAlias = tuple[int, Mapping[str, list[tree_sitter.Node]]]
CallbackType: TypeAlias = Callable[[str, MatchType, int], str]
