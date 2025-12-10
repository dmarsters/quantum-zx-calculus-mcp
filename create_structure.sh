#!/bin/bash
# Create directory structure and small files for quantum-zx-calculus-mcp server
# Usage: bash create_structure.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Creating directory structure for quantum-zx-calculus-mcp..."

# Core directories
mkdir -p src/quantum_zx
mkdir -p tests
mkdir -p docs
mkdir -p data

# Test subdirectories
mkdir -p tests/unit
mkdir -p tests/integration

# Documentation
mkdir -p docs/examples

echo "Creating small files..."

# .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
.pytest_cache/
.coverage
htmlcov/
.tox/
.venv
env/
venv/
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
.env
*.qasm
*.qpy
EOF

# __init__.py for src/quantum_zx
cat > src/quantum_zx/__init__.py << 'EOF'
"""
Quantum ZX-Calculus MCP Server

Deterministic mapping of quantum circuits to ZX-diagrams using categorical
quantum mechanics, with support for circuit simplification, optimization,
and educational visualization.

Based on Bob Coecke's categorical quantum mechanics framework and ZX-calculus.
"""

__version__ = "0.1.0"
__author__ = "Lushy Bricks"
EOF

# __init__.py for tests
cat > tests/__init__.py << 'EOF'
"""Tests for quantum-zx-calculus-mcp"""
EOF

# tests/unit/__init__.py
cat > tests/unit/__init__.py << 'EOF'
"""Unit tests for quantum ZX modules"""
EOF

# tests/integration/__init__.py
cat > tests/integration/__init__.py << 'EOF'
"""Integration tests for quantum ZX MCP server"""
EOF

# __main__.py for local testing
cat > src/quantum_zx/__main__.py << 'EOF'
"""
Local execution script for Quantum ZX-Calculus MCP server.

Usage:
    python -m quantum_zx

This runs the server locally for testing and development.
For production, use FastMCP Cloud deployment.
"""

from .server import mcp

if __name__ == "__main__":
    mcp.run()
EOF

# handler.py for FastMCP Cloud
cat > src/quantum_zx/handler.py << 'EOF'
"""
FastMCP Cloud entry point for Quantum ZX-Calculus server.

For FastMCP Cloud deployment, the entry point function must RETURN the server object.
The cloud platform handles the event loop and server.run() call.
"""

from .server import mcp

def handler():
    """Entry point for FastMCP Cloud deployment."""
    return mcp
EOF

# README.md
cat > README.md << 'EOF'
# Quantum ZX-Calculus MCP Server

Transform quantum circuits into ZX-diagrams using categorical quantum mechanics.

## Features

- **Circuit to ZX-Diagram Conversion**: Convert QASM and Qiskit circuits to ZX-diagram format
- **Deterministic Gate Taxonomy**: Complete mapping of quantum gates to ZX spiders
- **Simplification Strategies**: Layer 2 rewrite rule composition
- **Educational Visualization**: Generate markdown/SVG visualizations
- **Cost-Optimized**: 60%+ cost savings through deterministic mapping + single LLM synthesis

## Architecture

**Layer 1 (Foundation):** Gate taxonomy, spider types, phase parameters
**Layer 2 (Structure):** Rewrite rules, simplification strategies (deterministic)
**Layer 3 (Relational):** Circuit analysis, optimization selection
**Layer 4 (Contextual):** Claude synthesis for explanation and custom optimization

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
./tests/run_tests.sh
```

## Deployment

```bash
fastmcp deploy
```

## References

- Bob Coecke & Aleks Kissinger, "Picturing Quantum Processes" (Cambridge, 2017)
- PyZX: https://github.com/zxcalc/pyzx
- ZX-Calculus: https://zxcalculus.com/
EOF

# pyproject.toml - Standard FastMCP configuration
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "quantum-zx-calculus-mcp"
version = "0.1.0"
description = "Quantum ZX-Calculus MCP Server - Convert quantum circuits to ZX-diagrams using categorical quantum mechanics"
readme = "README.md"
requires-python = ">=3.10"
authors = [
    {name = "Lushy", email = "info@lushy.ai"}
]
license = {text = "MIT"}

dependencies = [
    "fastmcp>=0.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0",
    "ruff>=0.1.0",
]

[project.scripts]
quantum-zx = "quantum_zx.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/quantum_zx"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
asyncio_mode = "auto"

[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
target-version = "py310"
EOF

# STRUCTURE_PATTERN.md - Pattern documentation
cat > STRUCTURE_PATTERN.md << 'EOF'
# Quantum ZX-Calculus MCP - Standard Structure Pattern

This project follows the **Standard MCP Server Setup Pattern** established by Lushy.

## Phase 1: Automated Structure Generation

Run `create_structure.sh` to generate:
- Directory hierarchy (src/, tests/, docs/, data/)
- __init__.py files
- .gitignore
- __main__.py (local execution)
- handler.py (FastMCP Cloud)
- README.md
- pyproject.toml
- This documentation

## Phase 2: Manual Large File Placement

Copy these files to the project root:
- `quantum_zx_ologs.py` - Taxonomy definitions (gates, spiders, rules)
- `quantum_zx_calculus.py` - Core ZX-calculus implementation
- `quantum_zx_server.py` - FastMCP server definition
- `test_quantum_zx.py` - Comprehensive tests

## Phase 3: Verification

Run `verify_structure.sh` to validate:
1. Directory structure complete
2. All required files present
3. Import paths correct
4. Dependencies available

## Installation & Testing

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
./tests/run_tests.sh

# Run locally
python -m quantum_zx
```

## FastMCP Cloud Deployment

Entry point: `quantum_zx/handler.py:handler`

The handler function returns the MCP server object. FastMCP Cloud handles
the event loop and server.run() call.

## Cost Optimization Strategy

- **Layer 1:** Deterministic gate taxonomy (0 tokens)
- **Layer 2:** Rewrite rule composition (0 tokens)
- **Layer 3:** Circuit analysis (deterministic mapping, 0 tokens)
- **Layer 4:** Claude synthesis (single call, ~200-400 tokens)

**Result:** 60-85% cost savings vs pure LLM approach
EOF

echo "✓ Directory structure created"
echo "✓ Small files generated"
echo ""
echo "Next steps:"
echo "1. Copy large Python files (quantum_zx_ologs.py, quantum_zx_calculus.py, etc.)"
echo "2. Run: bash verify_structure.sh"
echo "3. Run: pip install -e \".[dev]\""
echo "4. Run: ./tests/run_tests.sh"
EOF
