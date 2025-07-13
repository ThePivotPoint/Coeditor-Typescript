"""
TypeScript scoped changes analysis module.
"""

from dataclasses import dataclass
from typing import *

import tree_sitter

from .common import *

@dataclass
class TsChangeScope:
    """A change scope for TypeScript code."""
    path: TsProjectPath
    tree: TsScopeTree
    spans: Sequence["TsStatementSpan"]
    subscopes: Mapping[str, "TsChangeScope"]
    parent_scope: "TsChangeScope | None"

    @staticmethod
    def from_tree(path: TsProjectPath, tree: TsScopeTree) -> "TsChangeScope":
        # Placeholder implementation
        return TsChangeScope(path, tree, [], {}, None)

@dataclass
class TsStatementSpan:
    """A statement span in TypeScript code."""
    nth_in_parent: int
    statements: Sequence[TsNode]
    scope: TsChangeScope 