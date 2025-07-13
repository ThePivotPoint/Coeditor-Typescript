#!/usr/bin/env python3
"""
Test for TypeScript version of prepare_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_ts_prepare_data_import():
    """Test TypeScript prepare_data imports work."""
    try:
        from prepare_data_ts import (
            TsC3ProblemChangeInlining,
            TsC3CombinedEncoder,
            make_or_load_ts_dataset,
            make_or_load_ts_transformed_dataset
        )
        print("✓ TypeScript prepare_data imports successful")
        
        # Test instantiation
        inlining = TsC3ProblemChangeInlining()
        encoder = TsC3CombinedEncoder(problem_tranform=inlining)
        
        print("✓ TypeScript prepare_data instantiation successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_ts_prepare_data_functions():
    """Test TypeScript prepare_data functions work."""
    try:
        from prepare_data_ts import (
            make_or_load_ts_dataset,
            make_or_load_ts_transformed_dataset,
            TsC3CombinedEncoder,
            TsC3ProblemChangeInlining
        )
        
        # Test dataset functions
        encoder = TsC3CombinedEncoder(
            problem_tranform=TsC3ProblemChangeInlining()
        )
        
        problems = make_or_load_ts_dataset(
            "test_ts_dataset",
            encoder.change_processor,
            ("train",),
            remake_problems=False
        )
        
        transformed = make_or_load_ts_transformed_dataset(
            "test_ts_dataset",
            problems,
            encoder
        )
        
        print("✓ TypeScript prepare_data functions work")
        return True
    except Exception as e:
        print(f"✗ Error in functions: {e}")
        return False

if __name__ == "__main__":
    print("Testing TypeScript prepare_data module...")
    
    tests = [
        test_ts_prepare_data_import,
        test_ts_prepare_data_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All TypeScript prepare_data tests passed!")
    else:
        print("✗ Some tests failed.")
        sys.exit(1) 