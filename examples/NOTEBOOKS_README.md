# Example Notebooks

Interactive demonstrations of the Quantum ZX-Calculus MCP Server showing the 4-layer categorical architecture and cost optimization in action.

## Quick Start (5 minutes)

**File:** `bell_state_quickstart.ipynb`

**Purpose:** Fast demonstration of core concepts using a single Bell state circuit.

**Shows:**
- All 4 layers in operation
- Zero-token deterministic operations (Layers 1-3)
- Cost comparison vs. pure LLM approach (70%+ savings)
- Connection to Breiner et al. categorical process planning

**Best for:**
- First-time users
- Quick demos to colleagues/investors
- Email attachments to academics
- Understanding the cost optimization model

**Running:**
```bash
jupyter notebook bell_state_quickstart.ipynb
```

---

## Complete Tutorial (20-30 minutes)

**File:** `zx_calculus_tutorial.ipynb`

**Purpose:** Comprehensive walkthrough with multiple circuits and optimization strategies.

**Shows:**
- Layer 1: Gate taxonomy and ZX representations
- Layer 2: Circuit parsing and ZX-diagram conversion
- Layer 3: Analysis and strategy selection
- Layer 4: Educational synthesis preparation
- Categorical properties (composition, type safety, functors)
- Multiple circuit examples (Bell state, T-count optimization, QFT fragment)
- Cost analysis across all operations

**Best for:**
- Learning ZX-calculus fundamentals
- Understanding categorical structure deeply
- Testing different optimization strategies
- Academic presentations/workshops
- Onboarding new developers

**Running:**
```bash
jupyter notebook zx_calculus_tutorial.ipynb
```

---

## Prerequisites

### Installation

```bash
# Install the quantum ZX-calculus package
pip install -e ".[dev]"

# Install Jupyter
pip install jupyter

# Optional: Install visualization tools
pip install matplotlib numpy
```

### Python Version
- Python 3.8+
- All notebooks tested on Python 3.9

---

## Notebook Structure

Both notebooks follow the same pedagogical pattern:

1. **Setup:** Import modules, verify installation
2. **Layer-by-layer demonstration:** Show each layer's operations and costs
3. **Cost analysis:** Compare to pure LLM approaches
4. **Categorical properties:** Demonstrate SMC structure
5. **Connections:** Link to Breiner et al. and broader Lushy framework

All code cells are self-contained and runnable independently (after setup cell).

---

## Expected Outputs

### Quick Start Outputs

```
Input: QASM circuit string
Cost so far: 0 tokens

=== LAYER 2: Circuit Parsing ===
Qubits: 2
Gates: [('h', [0]), ('cx', [0, 1])]
Cost: 0 tokens (deterministic regex parsing)

=== LAYER 3: Circuit Analysis ===
Total gates: 2
Clifford gates: 2
T gates: 0
Entanglement score: 0.500
Cost: 0 tokens (deterministic counting + heuristics)

[... full ZX-diagram representation ...]

COST COMPARISON SUMMARY
Categorical Approach: 150 tokens
Pure LLM Approach: 600 tokens
SAVINGS: 75.0%
```

### Tutorial Outputs

- 6 complete circuit examples
- ZX-diagram visualizations (text-based)
- Strategy comparisons
- ~26 operations performed
- Total cost: 0 tokens (Layers 1-3 only)

---

## Customizing the Notebooks

### Adding Your Own Circuits

Replace the QASM string with your circuit:

```python
my_circuit_qasm = """
OPENQASM 2.0;
qreg q[3];
h q[0];
cx q[0], q[1];
t q[1];
cx q[1], q[2];
"""

circuit = parse_qasm_circuit(my_circuit_qasm)
analysis = analyze_circuit(circuit)
# ... continue with analysis
```

### Testing Different Strategies

```python
strategies = [
    "clifford_simplification",
    "t_count_reduction",
    "measurement_based",
    "educational"
]

for strat_name in strategies:
    strat = select_simplification_strategy(analysis, desired_outcome=strat_name)
    print(f"\n{strat_name}: {strat['rewrite_rules']}")
```

### Visualizing ZX-Diagrams

The notebooks provide text-based output. To generate visual diagrams:

```python
# Option 1: Use pyzx (external library)
import pyzx as zx
circuit = zx.Circuit.from_qasm(qasm_string)
zx.draw(circuit)

# Option 2: Generate custom SVG (implement based on architecture_diagram.svg pattern)
from quantum_zx_visualization import generate_svg_diagram
svg = generate_svg_diagram(zx_diagram)
```

---

## Educational Use Cases

### For Students
- Learn quantum computing through categorical lens
- Understand ZX-calculus rewrite rules
- See how compositional semantics work in practice
- Compare optimization strategies quantitatively

### For Researchers
- Prototype new rewrite rules
- Test circuit simplification algorithms
- Analyze resource requirements (T-count, depth)
- Validate categorical properties

### For Developers
- Understand the 4-layer architecture
- See cost optimization in practice
- Learn categorical composition patterns
- Apply pattern to other domains

---

## Troubleshooting

### Import Errors
```python
# If modules not found, adjust path:
import sys
sys.path.append('/path/to/quantum-zx-mcp')
```

### Missing Dependencies
```bash
# Install all requirements:
pip install -e ".[dev]"

# Or individually:
pip install dataclasses typing enum
```

### Notebook Won't Run
```bash
# Ensure Jupyter installed:
pip install jupyter notebook

# Start Jupyter:
jupyter notebook

# Navigate to .ipynb file and open
```

---

## Next Steps After Notebooks

1. **Explore the codebase:**
   - `quantum_zx_ologs.py`: Layer 1 taxonomy
   - `quantum_zx_calculus.py`: Layers 2-3 operations
   - `quantum_zx_server.py`: MCP server integration

2. **Read the paper:**
   - Breiner et al., "Categorical Models for Process Planning" (2019)
   - See how manufacturing model maps to quantum circuits

3. **Try other Lushy domains:**
   - Heraldic blazonry MCP
   - Jazz improvisation MCP
   - See same 4-layer pattern across domains

4. **Contribute:**
   - Add new gates to taxonomy
   - Implement additional rewrite rules
   - Create visualization tools
   - Write additional example notebooks

---

## Academic Citation

If you use these notebooks in research or teaching:

```bibtex
@software{quantum_zx_notebooks,
  author = {Marsters, Dal},
  title = {Quantum ZX-Calculus Interactive Notebooks},
  year = {2025},
  url = {https://github.com/dmarsters/quantum-zx-calculus-mcp},
  note = {Educational notebooks demonstrating categorical circuit optimization}
}
```

---

**Academic collaboration welcome** - especially connections to:
- Applied category theory researchers
- Quantum computing educators
- Process planning / manufacturing modeling communities
