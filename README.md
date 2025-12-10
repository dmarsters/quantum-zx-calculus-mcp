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
