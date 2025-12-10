#!/bin/bash
# Verify structure and files for quantum-zx-calculus-mcp

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Verifying quantum-zx-calculus-mcp structure..."
echo ""

# Check directories
echo "Checking directories..."
REQUIRED_DIRS=(
    "src/quantum_zx"
    "tests"
    "docs"
    "data"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ $dir MISSING"
        exit 1
    fi
done

# Check key files
echo ""
echo "Checking files..."
REQUIRED_FILES=(
    "pyproject.toml"
    "README.md"
    "quantum_zx_ologs.py"
    "quantum_zx_calculus.py"
    "quantum_zx_server.py"
    "test_quantum_zx.py"
    "src/quantum_zx/__init__.py"
    "src/quantum_zx/__main__.py"
    "src/quantum_zx/handler.py"
    "tests/run_tests.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file MISSING"
        exit 1
    fi
done

# Check Python syntax
echo ""
echo "Checking Python syntax..."
python -m py_compile quantum_zx_ologs.py
echo "  ✓ quantum_zx_ologs.py"
python -m py_compile quantum_zx_calculus.py
echo "  ✓ quantum_zx_calculus.py"
python -m py_compile quantum_zx_server.py
echo "  ✓ quantum_zx_server.py"
python -m py_compile test_quantum_zx.py
echo "  ✓ test_quantum_zx.py"

# Check imports
echo ""
echo "Checking imports..."
python -c "from quantum_zx_ologs import QUANTUM_GATE_TAXONOMY" && echo "  ✓ quantum_zx_ologs imports"
python -c "from quantum_zx_calculus import parse_qasm_circuit" && echo "  ✓ quantum_zx_calculus imports"
python -c "from quantum_zx_server import mcp" && echo "  ✓ quantum_zx_server imports"

echo ""
echo "✓ All verification checks passed!"
echo ""
echo "Next steps:"
echo "1. pip install -e \".[dev]\""
echo "2. ./tests/run_tests.sh"
echo "3. python -m quantum_zx (to run server locally)"
