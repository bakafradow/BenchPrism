from .base import BaseTask
from .code_repair import CodeRepair
from .code_summarization import CodeSummarization
from .code_translation import CodeTranslation
from .io_reasoning import IOReasoning
from .mcq_answering import MCQAnswering
from .tag_classification import TagClassification
from .test_generation import TestGeneration

__all__ = [
    'BaseTask',
    'CodeTranslation',
    'CodeRepair',
    'CodeSummarization',
    'IOReasoning',
    'MCQAnswering',
    'TagClassification',
    'TestGeneration',
]
