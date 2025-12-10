#!/bin/bash
# Run all tests for quantum-zx-calculus-mcp

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Running quantum-zx-calculus-mcp tests..."
echo ""

# Run pytest with verbose output
python -m pytest test_quantum_zx.py -v --tb=short

echo ""
echo "✓ All tests passed!"
