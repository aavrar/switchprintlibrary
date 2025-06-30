"""
Code-Switch Aware AI Library

A library for detecting, remembering, and adapting to multilingual code-switching patterns.
"""

from .utils import VERSION, AUTHOR

__version__ = VERSION
__author__ = AUTHOR

from .detection import (
    LanguageDetector, 
    SwitchPointDetector, 
    EnhancedCodeSwitchDetector,
    OptimizedCodeSwitchDetector,
    PhraseCluster,
    EnhancedDetectionResult,
    OptimizedResult
)
from .memory import ConversationMemory, ConversationEntry, EmbeddingGenerator
from .retrieval import SimilarityRetriever
from .interface import CLI

__all__ = [
    "LanguageDetector",
    "SwitchPointDetector",
    "EnhancedCodeSwitchDetector",
    "OptimizedCodeSwitchDetector",
    "PhraseCluster", 
    "EnhancedDetectionResult",
    "OptimizedResult",
    "ConversationMemory",
    "ConversationEntry",
    "EmbeddingGenerator",
    "SimilarityRetriever",
    "CLI"
]