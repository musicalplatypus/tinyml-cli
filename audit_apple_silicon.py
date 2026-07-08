#!/usr/bin/env python3
"""
Apple Silicon Architecture Audit Script
This script audits the environment for architecture compatibility issues on Apple Silicon.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and return result."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"✓ {description}: SUCCESS")
            return True, result.stdout
        else:
            print(f"✗ {description}: FAILED")
            print(f"Error: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"✗ {description}: TIMEOUT")
        return False, "Command timed out"
    except Exception as e:
        print(f"✗ {description}: ERROR - {e}")
        return False, str(e)

def check_homebrew_libomp():
    """Check if Homebrew's libomp.dylib is present at the expected location."""
    print("\n=== Checking Homebrew OpenMP Library ===")

    # Check if libomp.dylib exists in /opt/homebrew/lib/
    libomp_path = "/opt/homebrew/lib/libomp.dylib"
    if os.path.exists(libomp_path):
        print(f"✓ Found libomp.dylib at {libomp_path}")
        return True, f"Found libomp.dylib at {libomp_path}"
    else:
        print(f"✗ libomp.dylib not found at {libomp_path}")
        # Try to find it elsewhere
        try:
            result = subprocess.run(
                ["find", "/", "-name", "libomp.dylib", "-type", "f"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                print("Found libomp.dylib at alternative location:")
                for line in result.stdout.strip().split('\n'):
                    print(f"  {line}")
                return True, f"Found libomp.dylib at alternative locations:\n{result.stdout}"
            else:
                print("No libomp.dylib found on system")
                return False, "libomp.dylib not found"
        except Exception as e:
            print(f"Error searching for libomp.dylib: {e}")
            return False, str(e)

def check_python_packages():
    """List and analyze Python packages with compiled extensions."""
    print("\n=== Checking Python Packages ===")

    # Get list of installed packages
    success, output = run_command("pip list --format=freeze", "Getting package list")
    if not success:
        return False, "Failed to get package list"

    # Filter for relevant packages
    packages = []
    for line in output.split('\n'):
        if line.strip() and not line.startswith('#') and 'numpy' in line or 'pandas' in line or 'torch' in line or 'opencv' in line:
            packages.append(line.split('==')[0])

    print(f"Found relevant packages: {packages}")

    # Check each package's .so files
    problematic_files = []

    for pkg in packages:
        try:
            import importlib.util
            spec = importlib.util.find_spec(pkg)
            if spec and spec.origin:
                pkg_path = Path(spec.origin).parent
                print(f"Package {pkg} found at: {pkg_path}")

                # Look for .so files
                so_files = list(pkg_path.rglob("*.so"))
                if so_files:
                    print(f"  Found .so files in {pkg_path}:")
                    for so_file in so_files:
                        print(f"    {so_file}")
                        success, arch_output = run_command(f"otool -f {so_file}", f"Checking architecture of {so_file}")
                        if success and "arm64" in arch_output.lower():
                            print(f"      ✓ Correctly architecture (arm64)")
                        elif success:
                            print(f"      ⚠ Architecture may be incorrect")
                            problematic_files.append(so_file)
                        else:
                            print(f"      ? Could not determine architecture")
                else:
                    print(f"  No .so files found in {pkg_path}")
        except Exception as e:
            print(f"Error checking package {pkg}: {e}")

    if problematic_files:
        return False, f"Found potentially problematic .so files: {problematic_files}"
    else:
        return True, "All packages checked successfully"

def check_system_info():
    """Check system information relevant to Apple Silicon."""
    print("\n=== Checking System Information ===")

    # Check macOS version
    success, output = run_command("sw_vers", "Getting macOS version")
    if not success:
        return False, "Failed to get macOS version"

    # Check architecture
    success, output = run_command("uname -m", "Checking system architecture")
    if not success:
        return False, "Failed to get system architecture"

    print(f"System architecture: {output.strip()}")

    # Check for Metal support
    success, output = run_command("system_profiler SPDisplaysDataType | grep -i metal", "Checking Metal support")
    if success:
        print("Metal support detected")
    else:
        print("Metal support check failed")

    return True, "System information collected"

def main():
    """Main audit function."""
    print("Apple Silicon Architecture Audit")
    print("=" * 40)

    issues = []

    # Run checks
    success, result = check_homebrew_libomp()
    if not success:
        issues.append(f"Homebrew OpenMP: {result}")

    success, result = check_python_packages()
    if not success:
        issues.append(f"Python packages: {result}")

    success, result = check_system_info()
    if not success:
        issues.append(f"System info: {result}")

    print("\n" + "=" * 40)
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("✓ All checks passed - no architecture compatibility issues detected")
        return 0

if __name__ == "__main__":
    sys.exit(main())