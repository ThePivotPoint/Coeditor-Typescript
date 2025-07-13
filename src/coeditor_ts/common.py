"""
Common types and utilities for TypeScript code analysis.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import *

# Import tree-sitter and TypeScript language
import tree_sitter
import tree_sitter_typescript

# Import common utilities from the main coeditor module
from coeditor.common import *
from coeditor._utils import rec_iter_files

# TypeScript-specific type definitions
TsNode = tree_sitter.Node
TsTree = tree_sitter.Tree
TsParser = tree_sitter.Parser

# TypeScript scope types
TsScopeTree = tree_sitter.Node  # FunctionDeclaration, ClassDeclaration, Program, etc.

# TypeScript full name type
TsFullName = NewType("TsFullName", str)

# TypeScript module name type
TsModuleName = NewType("TsModuleName", str)

# TypeScript project path type
TsProjectPath = NewType("TsProjectPath", str)

# Default ignore directories for TypeScript projects
TsDefaultIgnoreDirs = {
    ".git", "node_modules", "dist", "build", ".next", 
    ".nuxt", ".output", "coverage", ".nyc_output",
    "*.min.js", "*.bundle.js", "*.chunk.js"
}

@dataclass
class TsModule:
    """A light wrapper around a TypeScript module."""
    mname: TsModuleName
    tree: TsTree
    file_path: Path

    @property
    def as_scope(self) -> "TsChangeScope":
        from .scoped_changes import TsChangeScope
        return TsChangeScope.from_tree(TsProjectPath(str(self.mname)), self.tree.root_node)

    @property
    def imported_names(self) -> set[str]:
        """Get all imported names from this TypeScript module."""
        names = set[str]()
        
        # Query for import statements
        query = self.tree.language.query("""
            (import_statement
                (import_clause
                    (named_imports
                        (import_specifier
                            (identifier) @import_name))) @import)
            (import_statement
                (import_clause
                    (identifier) @import_default) @import)
            (import_statement
                (import_clause
                    (namespace_import
                        (identifier) @import_namespace) @import))
        """)
        
        captures = query.captures(self.tree.root_node)
        for node, capture_name in captures:
            if capture_name in ["import_name", "import_default", "import_namespace"]:
                names.add(node.text.decode('utf-8'))
        
        return names

def get_typescript_files(project: Path) -> list[RelPath]:
    """Get all TypeScript files in the project."""
    ts_files = []
    for pattern in ["*.ts", "*.tsx"]:
        ts_files.extend(rec_iter_files(project, lambda p: p.name.endswith(pattern[1:])))
    return ts_files

def ts_path_to_module_name(path: RelPath) -> TsModuleName:
    """Convert a TypeScript file path to a module name."""
    # Remove .ts or .tsx extension
    module_path = path.with_suffix("")
    # Convert path separators to dots
    module_name = module_path.as_posix().replace("/", ".")
    return TsModuleName(module_name)

def parse_typescript_module(path: Path) -> tuple[TsModule, TsParser]:
    """Parse a TypeScript file and return the module and parser."""
    parser = TsParser()
    parser.language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    
    with open(path, 'rb') as f:
        code = f.read()
    
    tree = parser.parse(code)
    module_name = ts_path_to_module_name(to_rel_path(path.relative_to(path.parent)))
    
    return TsModule(module_name, tree, path), parser

def code_to_ts_module(code: str) -> TsTree:
    """Parse TypeScript code string and return the syntax tree."""
    parser = TsParser()
    parser.language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    return parser.parse(code.encode('utf-8')) 