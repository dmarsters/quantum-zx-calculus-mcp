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
