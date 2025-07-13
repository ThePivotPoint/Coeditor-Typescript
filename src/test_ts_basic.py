#!/usr/bin/env python3
"""
Basic test for TypeScript analysis module.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports work."""
    try:
        import tree_sitter
        import tree_sitter_typescript
        print("✓ tree-sitter imports successful")
        
        # Test language loading
        parser = tree_sitter.Parser()
        parser.language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
        print("✓ TypeScript language loaded successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_ts_module_creation():
    """Test TypeScript module creation."""
    try:
        from coeditor_ts.common import TsModule, parse_typescript_module
        
        # Create a simple TypeScript code
        ts_code = """
function hello(name: string): string {
    return `Hello, ${name}!`;
}

class Greeter {
    constructor(private name: string) {}
    
    greet(): string {
        return hello(this.name);
    }
}
"""
        
        # Parse the code
        import tree_sitter
        import tree_sitter_typescript
        
        parser = tree_sitter.Parser()
        parser.language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
        tree = parser.parse(ts_code.encode('utf-8'))
        
        # Create module
        from coeditor_ts.common import TsModuleName
        module = TsModule(TsModuleName("test"), tree, Path("test.ts"))
        print("✓ TypeScript module creation successful")
        
        return True
    except Exception as e:
        print(f"✗ Error in module creation: {e}")
        return False

def test_file_discovery():
    """Test TypeScript file discovery."""
    try:
        from coeditor_ts.common import get_typescript_files
        
        # Test with current directory
        current_dir = Path(__file__).parent
        ts_files = get_typescript_files(current_dir)
        print(f"✓ Found {len(ts_files)} TypeScript files in current directory")
        
        return True
    except Exception as e:
        print(f"✗ Error in file discovery: {e}")
        return False

def test_c3problem_import():
    try:
        from coeditor_ts.c3problem import (
            TsC3Problem, TsC3ProblemGenerator, TsC3ProblemTokenizer, TsUsageAnalyzer, TsDefinition
        )
        from coeditor_ts.common import TsFullName
        # 实例化
        _ = TsC3Problem(span=None, edit_line_ids=[], relevant_changes=[], relevant_unchanged={}, change_type=None, src_info={})
        _ = TsC3ProblemGenerator()
        _ = TsC3ProblemTokenizer()
        _ = TsUsageAnalyzer()
        _ = TsDefinition(full_name=TsFullName("test"))
        print("✓ coeditor_ts.c3problem import and instantiation successful")
        return True
    except Exception as e:
        print(f"✗ Error in c3problem import/instantiation: {e}")
        return False

def test_service_import():
    try:
        from coeditor_ts.service import TsChangeDetector
        from pathlib import Path
        _ = TsChangeDetector(project=Path("."))
        print("✓ coeditor_ts.service import and instantiation successful")
        return True
    except Exception as e:
        print(f"✗ Error in service import/instantiation: {e}")
        return False

if __name__ == "__main__":
    print("Testing TypeScript analysis module...")
    
    tests = [
        test_basic_imports,
        test_ts_module_creation,
        test_file_discovery,
        test_c3problem_import,  # 新增
        test_service_import,  # 新增
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Step 1 completed successfully.")
    else:
        print("✗ Some tests failed. Please fix the issues before proceeding.")
        sys.exit(1) 