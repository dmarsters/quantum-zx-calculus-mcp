# Quantum ZX-Calculus MCP Server

**Categorical quantum circuit optimization using symmetric monoidal category theory**

Transform quantum circuits into ZX-diagrams with deterministic simplification strategies and educational visualizations. Built on the ZX-calculus framework (Coecke & Duncan, 2008-2017) with a cost-optimized architecture achieving 60-85% LLM cost reduction.

---

## Overview

The ZX-calculus is a **graphical language for quantum computing** based on symmetric monoidal categories (SMCs), where quantum circuits are represented as string diagrams with colored "spiders" (nodes) and wires. This MCP server provides:

1. **Circuit → ZX-diagram conversion** (QASM/Qiskit → spider notation)
2. **Deterministic simplification** (spider fusion, phase cancellation, Hadamard removal)
3. **Optimization strategy selection** (Clifford simplification, T-count reduction, measurement-based)
4. **Educational synthesis** (natural language explanations + SVG visualizations)

## Why Categorical?

The ZX-calculus exploits **compositional semantics**: quantum gates compose like morphisms in a category, with:
- **Serial composition**: Gate sequencing (type-safe: outputs must match inputs)
- **Parallel composition**: Tensor product (multi-qubit operations)
- **Rewrite rules**: Semantics-preserving transformations (proven sound + complete)

This categorical structure enables **deterministic optimizations** (0 LLM tokens) for 60-85% of the workflow, with LLM synthesis reserved for explanation and customization.

---

## Architecture: 4-Layer Categorical Hierarchy

### Layer 1: Foundation (Pure Taxonomy)
**Cost:** 0 tokens (static lookups)

```python
# Deterministic quantum gate → ZX spider mapping
QUANTUM_GATE_TAXONOMY = {
    "h": QuantumGate(primary_spider=H_BOX, phase=0.0, is_clifford=True),
    "t": QuantumGate(primary_spider=Z_SPIDER, phase=0.25, is_clifford=False),
    "cx": QuantumGate(primary_spider=Z_SPIDER, phase=0.0, is_clifford=True),
    # ... 15+ gates with ZX equivalents
}
```

Provides:
- Complete gate taxonomy (Clifford, Clifford+T, measurement operators)
- Spider types (Z/X/H) and phase parameters
- Basis specifications (computational, Hadamard)
- Spider fusion rules (bialgebra structure)

### Layer 2: Structure (Deterministic Operations)
**Cost:** 0 tokens (pure computation)

```python
# Parse circuit, analyze properties, select strategy
circuit = parse_qasm_circuit(qasm_code)           # 0 tokens
analysis = analyze_circuit(circuit)               # 0 tokens  
zx_diagram = circuit_to_zx_diagram(circuit)       # 0 tokens
strategy = select_simplification_strategy(...)    # 0 tokens
```

Provides:
- Circuit parsing (QASM → gate sequence)
- Property analysis (T-count, entanglement, Clifford depth)
- ZX-diagram construction (spiders + wires + phases)
- Rewrite strategy selection (based on circuit properties)

### Layer 3: Relational (Compositional Rules)
**Cost:** 0 tokens (categorical logic)

Implements ZX-calculus rewrite rules:
- **Spider fusion**: Adjacent same-color spiders merge (phases add mod 2π)
- **Phase cancellation**: Phases summing to 2π disappear
- **Hadamard removal**: H · H = identity
- **Color change**: H · Z · H = X (basis transformation)

### Layer 4: Contextual (LLM Synthesis)
**Cost:** ~100-200 tokens (single LLM call)

LLM synthesizes:
- Natural language circuit explanations
- Custom optimization strategies
- Educational step-by-step visualizations
- Domain-specific simplifications

---

## Cost Optimization

Traditional approach: **LLM for entire workflow** (~500-1000 tokens)
```
User query → LLM analyzes → LLM parses → LLM optimizes → LLM visualizes
```

Categorical approach: **Deterministic layers + targeted synthesis** (~100-200 tokens)
```
User query → Parse (0) → Analyze (0) → ZX convert (0) → Select strategy (0) → LLM synthesizes explanation (100-200)
```

**Result:** 60-85% cost reduction by encoding domain expertise as categorical structure rather than prompt engineering.

---

## Features

### Circuit Conversion
- **QASM 2.0 support**: Standard quantum assembly language
- **Qiskit compatibility**: Python quantum framework integration
- **Gate coverage**: 15+ gates (Pauli, Clifford, T, CNOT, CZ, SWAP)
- **Multi-qubit circuits**: Arbitrary qubit counts with entanglement tracking

### Simplification Strategies
```python
# Automatically select optimization based on circuit analysis
strategy = select_simplification_strategy(
    analysis=circuit_analysis,
    desired_outcome="t_count_reduction"  # or "clifford_simplification", "educational"
)
```

Available strategies:
- **Clifford simplification**: Reduce Clifford gate count (stabilizer circuits)
- **T-count reduction**: Minimize T gates (critical for fault tolerance)
- **Measurement-based**: Convert to measurement-based quantum computation
- **Educational**: Step-by-step with explanations at each transformation

### Analysis & Metrics
```python
stats = get_circuit_statistics(circuit)
# Returns:
# - Gate composition (Clifford vs T counts)
# - Entanglement score (0-1 estimate)
# - Clifford depth, estimated T-count
# - Recommended optimization strategies
```

### Educational Visualization
- SVG string diagram generation
- Step-by-step rewrite sequences
- Natural language explanations of each transformation
- LaTeX-ready mathematical notation

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/quantum-zx-mcp.git
cd quantum-zx-mcp

# Install with development dependencies
pip install -e ".[dev]"

# Run tests
./tests/run_tests.sh
```

### Deploy to FastMCP Cloud

```bash
fastmcp deploy
```

---

## Quick Start

### Example 1: Simple Circuit Analysis

```python
from quantum_zx_calculus import parse_qasm_circuit, analyze_circuit

qasm = """
OPENQASM 2.0;
qreg q[2];
h q[0];
cx q[0], q[1];
"""

circuit = parse_qasm_circuit(qasm)
analysis = analyze_circuit(circuit)

print(f"Total gates: {analysis.total_gates}")
print(f"Clifford gates: {analysis.clifford_gates}")
print(f"Is Clifford-only: {analysis.is_clifford_only}")
print(f"Recommended strategies: {[s.value for s in analysis.recommended_strategies]}")
```

**Output:**
```
Total gates: 2
Clifford gates: 2
Is Clifford-only: True
Recommended strategies: ['clifford_simplification', 'educational']
```

### Example 2: ZX-Diagram Conversion

```python
from quantum_zx_calculus import circuit_to_zx_diagram

zx_diagram = circuit_to_zx_diagram(circuit)

print(f"Spiders: {len(zx_diagram.spiders)}")
print(f"Wires: {len(zx_diagram.wires)}")
print(f"Phases: {zx_diagram.phases}")
```

**Output:**
```
Spiders: 4
Wires: 3
Phases: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
```

### Example 3: Claude Synthesis (via MCP)

```
User: "Analyze this Bell state preparation circuit and explain what it does"

MCP Server:
1. Parses QASM → 2-qubit circuit (H, CNOT)
2. Converts to ZX → 4 spiders, 3 wires
3. Analyzes → Clifford-only, creates entanglement
4. Passes structured data to Claude

LLM (e.g., Claude) synthesizes:
"This circuit creates a Bell state (|Φ+⟩), a maximally entangled state. 
The Hadamard on qubit 0 creates superposition (|0⟩+|1⟩)/√2, then the CNOT 
entangles it with qubit 1. In ZX-calculus, this appears as a Z-spider 
(Hadamard) connected to an X-spider (CNOT control/target), forming the 
characteristic 'cup' structure of entanglement."
```

---

## Theoretical Foundation

# Structural Isomorphism: Manufacturing vs Quantum Circuits

## Categorical Correspondence Table

| **NIST Process Planning** | **ZX-Calculus MCP Server** | **Categorical Structure** |
|---------------------------|---------------------------|--------------------------|
| **Machine operations** | Quantum gates (H, T, CNOT, etc.) | Atomic morphisms in category |
| **Production line sequences** | Circuit gate sequences | Serial composition (type-safe) |
| **Parallel factory operations** | Multi-qubit tensor operations | Parallel composition (⊗) |
| **Task decomposition** | Gate → Spider conversion | Functorial mapping F: Gates → Spiders |
| **Process constraints** | Type matching (outputs → inputs) | Monoidal category coherence |
| **Resource routing** | Wire connectivity in ZX-diagram | String diagram topology |
| **Factory-level planning** | Optimization strategy selection | Higher-order morphism composition |
| **Supply chain coordination** | Educational synthesis & explanation | Contextual interpretation layer |
| **Process validation** | Semantics-preserving rewrites | Category-theoretic soundness proofs |
| **Cost optimization** | T-count reduction, Clifford simplification | Resource-aware functor selection |
| **Multi-scale hierarchy** | 4-layer architecture (Foundation → Contextual) | Compositional abstraction levels |
| **Identity processes** | Identity gates (bare wires) | Identity morphisms in category |

## Key Insight

Both systems exhibit **symmetric monoidal category (SMC) structure** where:

1. **Objects** = Resources/States (production line capacity ↔ qubit counts)
2. **Morphisms** = Processes/Gates (manufacturing tasks ↔ quantum operations)
3. **Composition** = Task sequencing (serial: ∘) and parallelization (parallel: ⊗)
4. **Rewrite rules** = Process optimization (semantics-preserving transformations)

## The Functorial Pattern

Both domains implement the same **hierarchical decomposition functor**:

```
High-level specification  ────F───→  Low-level operations
       ↓                                    ↓
   Sub-tasks                           Primitive actions
       ↓                                    ↓
  Validation                          Semantic preservation
```

Where **F preserves compositional structure** (F(g ∘ f) = F(g) ∘ F(f)) ensuring that:
- Factory plans decompose correctly into production line tasks
- Quantum circuits decompose correctly into spider diagrams
- Optimizations at any level preserve overall semantics

## Cost Optimization Parallel

**Manufacturing (Breiner et al.):**
- High-level planning (strategic decisions) → expensive computational resources
- Low-level execution (machine operations) → deterministic, fast
- **Strategy**: Formalize relationships to enable automated reasoning at lower levels

**Quantum Circuits (This MCP):**
- High-level synthesis (natural language explanations) → expensive LLM tokens
- Low-level operations (gate taxonomy, parsing) → deterministic, 0 tokens
- **Strategy**: Encode domain expertise categorically, reserve LLM for interpretation

Both achieve cost savings through **deterministic categorical layers** rather than recalculating everything from scratch at each level.

## Generalization Potential

This pattern applies to **any compositional domain**:
- **Manufacturing**: Tasks compose into workflows
- **Quantum computing**: Gates compose into circuits
- **Creative AI**: Aesthetic parameters compose into prompts
- **Software engineering**: Functions compose into programs
- **Biological systems**: Reactions compose into pathways

The common requirement: **compositional semantics** that can be formalized as a symmetric monoidal category.

### References

1. **Bob Coecke & Aleks Kissinger.** *Picturing Quantum Processes: A First Course in Quantum Theory and Diagrammatic Reasoning.* Cambridge University Press, 2017.
   - Foundational textbook on ZX-calculus and categorical quantum mechanics

2. **Bob Coecke & Ross Duncan.** "Interacting Quantum Observables: Categorical Algebra and Diagrammatics." *New Journal of Physics* 13.4 (2011): 043016.
   - Original ZX-calculus formulation with completeness proof

3. **Spencer Breiner, Albert Jones & Eswaran Subrahmanian.** "Categorical Models for Process Planning." *Computer Industry* 112 (2019).
   - Process hierarchy decomposition using symmetric monoidal categories
   - Theoretical validation for hierarchical categorical architectures

4. **PyZX Library**: https://github.com/zxcalc/pyzx
   - Python implementation of ZX-calculus (our taxonomy is compatible)

5. **ZX-Calculus Resource**: https://zxcalculus.com/
   - Community portal with interactive tutorials

### Categorical Composition Theory

This server demonstrates **functorial decomposition** of quantum circuits:

```
Circuit (high-level) ──────────────────> ZX-diagram (semantic)
    │                                           │
    │ Functor F                                 │ Functor G
    ↓                                           ↓
Gate sequence (syntax) ─────────────────> Spider graph (rewrite rules)
```

Where:
- **F**: Circuit parsing functor (syntax → structure)
- **G**: Optimization functor (semantics → strategies)
- **Composition**: G ∘ F preserves quantum semantics while enabling classical simplification

This pattern generalizes to **any domain** with compositional structure - see Lushy's framework of 50+ categorical MCP servers across diverse aesthetic domains.

---

## Use Cases

### Quantum Computing Education
- Teach ZX-calculus interactively with natural language explanations
- Visualize how quantum gates compose categorically
- Step-through circuit simplifications with mathematical justification

### Research & Development
- Rapid prototyping of quantum algorithms
- Circuit optimization with T-count minimization
- Measurement-based quantum computation experiments

### Fault-Tolerant Compilation
- Analyze resource requirements (T-count, Clifford depth)
- Optimize circuits for error correction codes
- Compare optimization strategies quantitatively

### Integration with AI Workflows
- Compositional semantics for multi-domain AI systems
- Deterministic computation layers for cost optimization
- Educational synthesis for domain knowledge transfer

---

## Relationship to Lushy.app Framework

This MCP server is part of the **Lushy categorical composition ecosystem**, which applies the same 4-layer architecture across 50+ domains:

**Common pattern:**
1. **Layer 1**: Deterministic domain taxonomy (0 tokens)
2. **Layer 2**: Compositional operations (0 tokens)
3. **Layer 3**: Strategy selection (0 tokens)
4. **Layer 4**: Claude synthesis (~100-200 tokens)

**Other Lushy domains:**
- Heraldic blazonry (visual vocabulary + tincture rules)
- Jazz improvisation (chord progressions + modal theory)
- Microscopy aesthetics (fluorescence channels + scale context)
- Nuclear explosion phases (fireball dynamics + atmospheric effects)

**Cross-domain composition:**
- Compatibility matrices determine which domains compose safely
- Graph-theoretic analysis reveals aesthetic clusters
- Frobenius ordering minimizes synthesis cost

---

## Contributing

We welcome contributions! Areas of interest:

1. **Gate coverage expansion**: Add parametric gates (Rx, Ry, Rz), controlled operations
2. **Additional rewrite rules**: Implement local complementation, phase teleportation
3. **Visualization improvements**: Interactive SVG with hover states, animation sequences
4. **Backend integrations**: Qiskit runtime, Cirq compatibility, QASM 3.0 support
5. **Educational content**: Tutorial notebooks, worked examples, proof walkthroughs

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest --cov=quantum_zx_calculus tests/

# Format code
black quantum_zx_calculus/
flake8 quantum_zx_calculus/

# Type checking
mypy quantum_zx_calculus/
```

---

## Citation

If you use this server in academic work, please cite:

```bibtex
@software{quantum_zx_mcp,
  author = {Marsters, Dal},
  title = {Quantum ZX-Calculus MCP Server: Categorical Circuit Optimization},
  year = {2025},
  url = {https://github.com/dmarsters/quantum-zx-calculus-mcp},
  note = {MCP server implementing ZX-calculus with hierarchical categorical architecture}
}
```

And the foundational ZX-calculus work:

```bibtex
@book{coecke2017picturing,
  title={Picturing Quantum Processes},
  author={Coecke, Bob and Kissinger, Aleks},
  year={2017},
  publisher={Cambridge University Press}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Contact

Dal Marsters  
Co-founder, Lushy  
dal@lushy.app  
https://www.linkedin.com/in/dalmarsters/

For questions about:
- **Categorical theory**: See theoretical foundation section
- **Implementation details**: Open a GitHub issue
- **Academic collaboration**: Email directly
- **Commercial applications**: Via Lushy website

---

## Acknowledgements

- **Bob Coecke & Aleks Kissinger**: ZX-calculus theoretical foundation
- **PyZX team**: Reference implementation and community resources
- **Spencer Breiner et al.**: Categorical process planning framework (NIST)
- **Applied Category Theory community**: Compositional semantics research
