"""
TypeScript code analysis module for CoEditor.
This module provides TypeScript-specific code parsing and analysis functionality.
"""

from .common import *
from .scoped_changes import *
from .c3problem import *
from .service import *

__all__ = [
    # Common types and utilities
    "TsModule", "TsChangeScope", "TsStatementSpan", "TsNode", "TsScopeTree",
    # Analysis classes
    "TsUsageAnalyzer", "TsDefinition", "TsFullName",
    # Problem generation
    "TsC3Problem", "TsC3ProblemGenerator", "TsC3ProblemTokenizer",
    # Service layer
    "TsChangeDetector",
] 