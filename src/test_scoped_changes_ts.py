"""
Test TypeScript scoped changes analysis.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from coeditor_ts.scoped_changes import *
from coeditor_ts.common import *
from coeditor.change import Modified

def test_ts_change_scope_basic():
    """Test basic TypeScript change scope creation."""
    print("Testing TypeScript change scope basic functionality...")
    
    # Create a simple TypeScript code
    code = """
function hello(name: string): string {
    return `Hello, ${name}!`;
}

class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}
"""
    
    # Parse the code
    tree = code_to_ts_module(code)
    
    # Create change scope
    path = TsProjectPath("test.module")
    scope = TsChangeScope.from_tree(path, tree.root_node)
    
    print(f"Created scope: {scope}")
    print(f"Scope type: {scope.tree.type}")
    print(f"Number of spans: {len(scope.spans)}")
    print(f"Number of subscopes: {len(scope.subscopes)}")
    
    # Check that we have subscopes for function and class
    assert "hello" in scope.subscopes, "Function 'hello' should be in subscopes"
    assert "Calculator" in scope.subscopes, "Class 'Calculator' should be in subscopes"
    
    print("✓ Basic TypeScript change scope test passed!")

def test_ts_statement_span():
    """Test TypeScript statement span creation."""
    print("Testing TypeScript statement span...")
    
    # Create a simple TypeScript code with statements
    code = """
let x = 1;
let y = 2;
console.log(x + y);
"""
    
    # Parse the code
    tree = code_to_ts_module(code)
    
    # Create change scope
    path = TsProjectPath("test.module")
    scope = TsChangeScope.from_tree(path, tree.root_node)
    
    print(f"Created scope with {len(scope.spans)} spans")
    
    # Check that we have spans
    assert len(scope.spans) > 0, "Should have at least one span"
    
    for i, span in enumerate(scope.spans):
        print(f"Span {i}: {span}")
        print(f"  Code: {repr(span.code)}")
        print(f"  Line range: {span.line_range}")
    
    print("✓ TypeScript statement span test passed!")

def test_ts_module_change():
    """Test TypeScript module change creation."""
    print("Testing TypeScript module change...")
    
    # Create a simple TypeScript module
    code = """
export function greet(name: string): string {
    return `Hello, ${name}!`;
}
"""
    
    # Parse the code
    tree = code_to_ts_module(code)
    module = TsModule(TsModuleName("test"), tree, Path("test.ts"))
    
    # Create module change
    module_change = Modified(module, module)  # No change for now
    module_change_obj = TsModuleChange.from_modules(module_change)
    
    print(f"Created module change: {module_change_obj}")
    print(f"Number of changes: {len(module_change_obj.changed)}")
    
    print("✓ TypeScript module change test passed!")

def test_ts_file_parsing():
    """Test TypeScript file parsing."""
    print("Testing TypeScript file parsing...")
    
    # Create a temporary TypeScript file
    test_file = Path("test_file.ts")
    test_code = """
interface User {
    id: number;
    name: string;
}

function createUser(id: number, name: string): User {
    return { id, name };
}

export { createUser };
"""
    
    try:
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Parse the file
        module, parser = parse_typescript_module_script(Path("."), test_file)
        
        print(f"Parsed module: {module}")
        print(f"Module name: {module.mname}")
        print(f"Tree type: {module.tree.root_node.type}")
        
        # Create change scope
        scope = module.as_scope
        print(f"Created scope: {scope}")
        print(f"Number of subscopes: {len(scope.subscopes)}")
        
        # Check imported names
        imported_names = module.imported_names
        print(f"Imported names: {imported_names}")
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
    
    print("✓ TypeScript file parsing test passed!")

if __name__ == "__main__":
    print("Running TypeScript scoped changes tests...")
    
    try:
        test_ts_change_scope_basic()
        test_ts_statement_span()
        test_ts_module_change()
        test_ts_file_parsing()
        
        print("\n🎉 All TypeScript scoped changes tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 