from collections.abc import Mapping
from functools import lru_cache, singledispatch
from itertools import product, tee
from typing import Any

import jpype as jp
import jpype.imports

from org.example.parser.common.factory import MyParserFactory
from org.example.styler import Styler
from org.example.styler.arrangement.modifier import ModifierOrderStyler
from org.example.styler.arrangement.modifier.style import (
    ModifierOrderProperty, ModifierOrderStyle)
from org.example.styler.declaration.layout import DeclarationLayoutStyler
from org.example.styler.declaration.layout.style import (
    DeclarationLayoutProperty, DeclarationLayoutStyle)
from org.example.styler.format.indention import IndentionStyler
from org.example.styler.format.indention.style import (IndentionProperty,
                                                       IndentionStyle)
from org.example.styler.format.newline import NewlineStyler
from org.example.styler.format.newline.bodylayout import (BodyLayoutStyler,
                                                          BodySizeType,
                                                          BodyTypeEnum)
from org.example.styler.format.newline.bodylayout.style import (
    BodyContext, BodyLayoutProperty, BodyLayoutStyle)
from org.example.styler.format.newline.inter import InterNewlineStyler
from org.example.styler.format.newline.inter.style import (
    InterNewlineProperty, InterNewlineStyle)
from org.example.styler.format.newline.intra import IntraNewlineStyler
from org.example.styler.format.newline.intra.style import (
    IntraNewlineContext, IntraNewlineProperty, IntraNewlineStyle)
from org.example.styler.format.newline.style import (NewlineContext,
                                                     NewlineProperty,
                                                     NewlineStyle)
from org.example.styler.format.space import SpaceStyler
from org.example.styler.format.space.style import (SpaceContext, SpaceProperty,
                                                   SpaceStyle)
from org.example.styler.ifelse.bodyorder import IfElseBodyOrderStyler
from org.example.styler.ifelse.bodyorder.style import (IfElseBodyOrderProperty,
                                                       IfElseBodyOrderStyle)
from org.example.styler.naming import MyCaseFormat, NameType
from org.example.styler.naming.format import NamingStyler, SymbolAttr
from org.example.styler.naming.format.style import (NamingFormatContext,
                                                    NamingFormatProperty,
                                                    NamingFormatStyle)
from org.example.styler.optionalbrace import OptionalBraceStyler
from org.example.styler.optionalbrace.style import (OptionalBraceContext,
                                                    OptionalBraceProperty,
                                                    OptionalBraceStyle)
from org.example.styler.structure import (EquivalentStructure,
                                          EquivalentStructureManager,
                                          StructureStyler)
from org.example.styler.structure.style import (StructPreferenceContext,
                                                StructPreferenceProperty,
                                                StructureStyle)


@singledispatch
def build_styler(styler: Styler, lang: str, choices: Mapping[str, Any]) -> None:
  """
  Builds a StyleX styler instance based on given choices.

  @param styler: The Styler instance to configure.
  @param lang: The language for which the styler is being built.
  @param choices: A *dict* of user-defined choices.
  """
  raise NotImplementedError(f'Styler builder not implemented for {type(styler)}')


@build_styler.register
def _(styler: IndentionStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = IndentionStyle()
  match choices['indention_unit']:
    case 'TAB':
      prop = IndentionProperty(1, '\t', False, 0)
    case 'TWO_SPACES':
      prop = IndentionProperty(2, ' ', False, 0)
    case 'FOUR_SPACES':
      prop = IndentionProperty(4, ' ', False, 0)
    case _:
      raise ValueError(f'Unknown indention unit choice: {choices["indention_unit"]}')
  style.addRule(None, prop)
  styler.setStyle(style)


@build_styler.register
def _(styler: SpaceStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = SpaceStyle()
  prop_dual = SpaceProperty(True, True)
  prop_mono = SpaceProperty(False, True)
  prop_none = SpaceProperty(False, False)

  def set_spacing(spacing: bool, ltoken: str, rtoken: str = '') -> None:
    context = SpaceContext(ltoken, rtoken)
    if spacing:
      if not rtoken:
        style.addRule(context, prop_dual)
      else:
        style.addRule(context, prop_mono)
    else:
      style.addRule(context, prop_none)

  set_spacing(choices['operator_spacing'], 'BIN_OP')
  set_spacing(choices['operator_spacing'], 'QUESTION')
  set_spacing(choices['operator_spacing'], 'COLON')
  set_spacing(choices['operator_spacing'], 'COLON_COLON')
  set_spacing(choices['call_spacing'], 'IDENTIFIER', '(')
  set_spacing(choices['call_spacing'], 'DOT')
  set_spacing(choices['comma_spacing'], ',', 'LITERAL')
  set_spacing(choices['comma_spacing'], ',', 'IDENTIFIER')
  set_spacing(choices['comma_spacing'], ',', 'UNARY_OP')
  set_spacing(choices['comma_spacing'], ',', 'KEYWORD')
  set_spacing(choices['comma_spacing'], ',', '(')
  set_spacing(choices['comma_spacing'], ',', ')')
  set_spacing(choices['comma_spacing'], ',', '[')
  set_spacing(choices['comma_spacing'], ',', ']')
  set_spacing(choices['comma_spacing'], ',', '{')
  set_spacing(choices['comma_spacing'], ',', '}')
  set_spacing(choices['semicolon_spacing'], 'LITERAL', ';')
  set_spacing(choices['semicolon_spacing'], 'IDENTIFIER', ';')
  set_spacing(choices['semicolon_spacing'], 'UNARY_OP', ';')
  set_spacing(choices['semicolon_spacing'], 'KEYWORD', ';')
  set_spacing(choices['semicolon_spacing'], ')', ';')
  set_spacing(choices['semicolon_spacing'], ']', ';')
  set_spacing(choices['semicolon_spacing'], '{', ';')
  set_spacing(choices['semicolon_spacing'], '}', ';')
  styler.setStyle(style)


@build_styler.register
def _(styler: NewlineStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = NewlineStyle()

  @lru_cache(maxsize=None)
  def get_prop(num_newlines: int) -> NewlineProperty:
    return NewlineProperty(num_newlines, 1)

  def set_newline(newline: bool, lnode: str, l_ast: bool, rnode: str, r_ast: bool,
                  extra_newline: int = 0) -> None:
    node_types = jp.java.util.List.of([NewlineContext.NodeType(lnode, l_ast),
                                       NewlineContext.NodeType(rnode, r_ast)])
    lengths = jp.java.util.List.of([jp.JDouble(0)])  # dummy lengths
    context = NewlineContext(node_types, lengths)
    if newline:
      style.addRule(context, get_prop(1 + extra_newline))
    else:
      style.addRule(context, get_prop(extra_newline))

  for (lnode, l_ast), (rnode, r_ast) in product(*tee(
    [('fieldDeclarationList', True), ('methodDeclarationList', True), ('methodDeclaration', True)],
  )):
    set_newline(choices['member_padding'], lnode, l_ast, rnode, r_ast, 1)
  for (lnode, l_ast), (rnode, r_ast) in product(*tee(
    [('BRANCH_STMT', True), ('LOOP_STMT', True), ('block', True)],
  )):
    set_newline(choices['block_padding'], lnode, l_ast, rnode, r_ast, 1)
  for (lnode, l_ast), (rnode, r_ast) in product(*tee(
    [('localVariableDeclarationStmt', True)]
  )):
    set_newline(choices['declaration_padding'], lnode, l_ast, rnode, r_ast)
  styler.setStyle(style)


@build_styler.register
def _(styler: IntraNewlineStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = IntraNewlineStyle()
  if choices['long_line_wrapping']:
    context = IntraNewlineContext(20)
    prop = IntraNewlineProperty(1, jp.java.util.List.of(['1']), 10)
    style.addRule(context, prop)
  styler.setStyle(style)


@build_styler.register
def _(styler: InterNewlineStyler, lang: str, choices: Mapping[str, Any]) -> None:
  ...


@build_styler.register
def _(styler: BodyLayoutStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = BodyLayoutStyle()
  if choices['break_before_brace']:
    prop = BodyLayoutProperty(True, True, True, True)
  else:
    prop = BodyLayoutProperty(False, True, True, True)
  for body_type, body_size, has_left_neighbor, has_right_neighbor, has_brace in product(
    ['DEC_BODY', 'STMT_BODY'],
    ['EMPTY', 'ONE_SINGLE_STMT', 'ONE_COMPOUND_STMT', 'MULTI_STMTS'],
    [True],
    [False, True],
    [True],
  ):
    context = BodyContext(BodyTypeEnum.valueOf(body_type),
                          BodySizeType.valueOf(body_size),
                          has_left_neighbor,
                          has_right_neighbor,
                          has_brace)
    style.addRule(context, prop)
  styler.setStyle(style)


@build_styler.register
def _(styler: OptionalBraceStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = OptionalBraceStyle()
  prop = OptionalBraceProperty(not choices['omit_braces'])
  for body_type, body_size, has_right_neighbor in product(
    ['STMT_BODY'],
    ['EMPTY', 'ONE_SINGLE_STMT'],
    [False, True],
  ):
    context = OptionalBraceContext(BodyTypeEnum.valueOf(body_type),
                                   BodySizeType.valueOf(body_size),
                                   has_right_neighbor)
    style.addRule(context, prop)
  styler.setStyle(style)


@build_styler.register
def _(styler: ModifierOrderStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = ModifierOrderStyle()
  modifiers = 'ANNOTATION ACCESS_CONTROL abstract static final' \
              'sealed non-sealed transient volatile default' \
              'synchronized native strictfp'.split()
  match choices['order']:
    case 'STANDARD':
      modifier_list = jp.java.util.List.of(modifiers)
    case 'REVERSED':
      modifier_list = jp.java.util.List.of(list(reversed(modifiers)))
    case _:
      raise ValueError(f'Unknown modifier order choice: {choices["order"]}')
  prop = ModifierOrderProperty(modifier_list)
  style.addRule(None, prop)
  styler.setStyle(style)


@build_styler.register
def _(styler: DeclarationLayoutStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = DeclarationLayoutStyle()
  if choices['merge_declarations']:
    prop = DeclarationLayoutProperty(1., 1.)
  else:
    prop = DeclarationLayoutProperty(0., 1.)
  style.addRule(None, prop)
  styler.setStyle(style)


@build_styler.register
def _(styler: StructureStyler, lang: str, choices: Mapping[str, Any]) -> None:
  # implemented in Java's side due to complexity...
  choice_map = jp.java.util.Map.ofEntries([jp.java.util.Map.entry(jp.JInt(k), jp.JInt(v))
                                           for k, v in choices.items()])
  styler.setAs(lang, choice_map)


@build_styler.register
def _(styler: IfElseBodyOrderStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = IfElseBodyOrderStyle()
  if choices['short_body_comes_first']:
    prop = IfElseBodyOrderProperty(True)
  else:
    prop = IfElseBodyOrderProperty(False)
  style.addRule(None, prop)
  styler.setStyle(style)


@build_styler.register
def _(styler: NamingStyler, lang: str, choices: Mapping[str, Any]) -> None:
  style = NamingFormatStyle()
  prop = NamingFormatProperty(False, MyCaseFormat.valueOf(choices['case_format']),
                              4 if choices['brief'] else jp.java.lang.Integer.MAX_VALUE)
  for name_type, attr in product(
    ['LOCAL_VARIABLE'],
    [None, 'EXPLICIT_CONST', 'IMPLICIT_CONST']):
    context = NamingFormatContext(NameType.valueOf(name_type))
    if attr:
      context.addAttr(SymbolAttr.valueOf(attr))
    style.addRule(context, prop)
  styler.setStyle(style)
