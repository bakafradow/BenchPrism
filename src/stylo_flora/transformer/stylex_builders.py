from abc import ABC, abstractmethod
import jpype as jp


IndentionStyler = jp.JClass('org.example.styler.format.indention.IndentionStyler')
NewlineStyler = jp.JClass('org.example.styler.format.newline.NewlineStyler')
SpaceStyler = jp.JClass('org.example.styler.format.space.SpaceStyler')
EquivalentStructure = jp.JClass('org.example.styler.structure.EquivalentStructure')
EquivalentStructureManager = jp.JClass('org.example.styler.structure.EquivalentStructureManager')
StructureStyler = jp.JClass('org.example.styler.structure.StructureStyler')


class BaseBuilder(ABC):
  """
  Abstract base class for StyleX style builders.
  """

  @abstractmethod
  def build(self, options: dict) -> jp.JObject:
    """
    Builds and returns a StyleX style instance for corresponding styler.
    :return: the built StyleX style

    :param options: options for the styler
    """
    pass
