#!/bin/bash
set -e

# Script to build and validate mmcli documentation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building mmcli documentation..."

# Install sphinx if needed (check if available)
if ! command -v sphinx-build &> /dev/null; then
    echo "Installing Sphinx..."
    pip install --user "sphinx>=8.0,<8.2" --break-system-packages --quiet
fi

cd "$PROJECT_ROOT"

# Build HTML docs
echo "Building HTML documentation..."
sphinx-build -b html docs/ docs/_build/html

# Validate links (linkcheck builder)
echo "Checking documentation links..."
sphinx-build -b linkcheck docs/ docs/_build/linkcheck || true

echo "Documentation built successfully!"
echo "  HTML: docs/_build/html/index.html"
