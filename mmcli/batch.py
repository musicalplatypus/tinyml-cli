"""
Batch processing utilities for handling multiple projects or datasets.
"""

import os
import glob
from typing import List, Callable, Any


def expand_project_paths(patterns: List[str]) -> List[str]:
    """
    Expand glob patterns to list of paths.

    Args:
        patterns: List of file/directory patterns (can include globs)

    Returns:
        List of expanded paths
    """
    paths = []
    for pattern in patterns:
        # Handle both glob patterns and direct paths
        if '*' in pattern or '?' in pattern:
            matches = glob.glob(pattern, recursive=True)
            paths.extend(matches)
        else:
            paths.append(pattern)

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            result.append(p)

    return result


def run_batch(
    command_func: Callable,
    paths: List[str],
    **kwargs
) -> dict:
    """
    Run a command on multiple paths.

    Args:
        command_func: Function to call for each path
        paths: List of project/dataset paths
        **kwargs: Additional arguments passed to command_func

    Returns:
        Dictionary mapping paths to results/errors
    """
    results = {}

    for path in paths:
        try:
            result = command_func(path, **kwargs)
            results[path] = {"success": True, "result": result}
        except Exception as e:
            results[path] = {
                "success": False,
                "error": str(e),
                "traceback": None  # Can add traceback if needed
            }

    return results


def format_batch_results(results: dict) -> str:
    """
    Format batch results as human-readable output.

    Args:
        results: Dictionary from run_batch()

    Returns:
        Formatted string with summary and details
    """
    lines = []

    # Summary
    success_count = sum(1 for r in results.values() if r.get("success"))
    fail_count = len(results) - success_count

    lines.append("=" * 60)
    lines.append(f"BATCH PROCESSING RESULTS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total: {len(results)}")
    lines.append(f"Success: {success_count}")
    lines.append(f"Failed: {fail_count}")
    lines.append("")

    # Detailed results
    if success_count > 0:
        lines.append("SUCCESSFUL:")
        for path, result in results.items():
            if result.get("success"):
                lines.append(f"  ✓ {path}")
        lines.append("")

    if fail_count > 0:
        lines.append("FAILED:")
        for path, result in results.items():
            if not result.get("success"):
                error = result.get("error", "Unknown error")
                lines.append(f"  ✗ {path}: {error[:80]}")

    return "\n".join(lines)
