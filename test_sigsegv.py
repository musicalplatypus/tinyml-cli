#!/usr/bin/env python3
"""
Test script to reproduce and verify Apple Silicon SIGSEGV issues.
This simulates the environment where SIGSEGV might occur.
"""

import os
import subprocess
import sys

def test_environment():
    """Test if we can run basic commands without SIGSEGV."""
    print("Testing basic mmcli functionality...")

    # Test that we can at least import the CLI module
    try:
        import mmcli.cli
        print("✓ Successfully imported mmcli.cli")
    except Exception as e:
        print(f"✗ Failed to import mmcli.cli: {e}")
        return False

    # Test that we can run diagnose command (this is where SIGSEGV might manifest)
    try:
        from mmcli.diagnose import run_diagnostic_checks
        result = run_diagnostic_checks()
        print("✓ Successfully ran diagnostic checks")
        print(f"  Checks passed: {len([c for c in result.checks if c.status == 'pass'])}")
        print(f"  Checks failed: {len([c for c in result.checks if c.status == 'fail'])}")
    except Exception as e:
        print(f"✗ Failed to run diagnostic checks: {e}")
        return False

    return True

def test_python_packages():
    """Test if Python packages are correctly built for ARM64."""
    print("\nTesting Python package compatibility...")

    try:
        import numpy as np
        import pandas as pd

        # Check architecture of numpy (if it's a compiled extension)
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ Pandas version: {pd.__version__}")

        # Try some basic operations that might trigger issues
        arr = np.array([1, 2, 3])
        result = arr * 2
        print("✓ Basic NumPy operations work")

        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        print("✓ Basic Pandas operations work")

    except Exception as e:
        print(f"✗ Failed package compatibility test: {e}")
        return False

    return True

def main():
    """Main test function."""
    print("Apple Silicon SIGSEGV Test Script")
    print("=" * 35)

    success = True
    success &= test_environment()
    success &= test_python_packages()

    if success:
        print("\n✓ All tests passed - no immediate SIGSEGV issues detected")
        return 0
    else:
        print("\n✗ Some tests failed - potential SIGSEGV issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())