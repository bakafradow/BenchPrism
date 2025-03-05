from .. import MatchType


def increment_decrement_preferences(code: str, match: MatchType, dst: int) -> str:
  stmt_node = match[1]['stmt'][0]
  # skip when the parent node is function call or array access
  if (stmt_node.parent.type == 'argument_list'
      or stmt_node.parent.type == 'array_access'):
    return code[stmt_node.start_byte:stmt_node.end_byte]
  var_node = match[1]['var'][0]
  op_node = match[1]['op'][0]
  match dst:
    case 0:
      mutant = code[op_node.start_byte:op_node.end_byte] + code[var_node.start_byte:var_node.end_byte]
    case 1:
      mutant = code[op_node.start_byte:op_node.end_byte] + code[var_node.start_byte:var_node.end_byte]
    case 2:
      mutant = code[var_node.start_byte:var_node.end_byte] + ' ' + code[op_node.start_byte] + '= 1'
    case _:
      raise ValueError('Unimplemented.')
  return mutant
