"""
Example: Using quantum_zx_visualization

Shows how to create ZX-diagram visualizations from quantum circuits.
"""

# Make sure you have the modules in your path
import sys
sys.path.append('..')  # Adjust as needed

from quantum_zx_calculus import parse_qasm_circuit, circuit_to_zx_diagram
from quantum_zx_visualization import (
    generate_svg_diagram,
    save_svg_to_file,
    display_in_notebook
)

# Example 1: Bell State
print("=== Example 1: Bell State ===")

bell_qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

circuit = parse_qasm_circuit(bell_qasm)
zx_diagram = circuit_to_zx_diagram(circuit)

# Generate SVG
svg = generate_svg_diagram(zx_diagram)

# Save to file
save_svg_to_file(svg, "bell_state_zx.svg")

# Display in notebook (if running in Jupyter)
try:
    display_in_notebook(svg)
    print("Displayed in notebook")
except:
    print("SVG saved to file (not in notebook environment)")

# Example 2: T-gate circuit
print("\n=== Example 2: T-gate Circuit ===")

t_circuit_qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
t q[0];
cx q[0], q[1];
t q[1];
"""

circuit2 = parse_qasm_circuit(t_circuit_qasm)
zx_diagram2 = circuit_to_zx_diagram(circuit2)

# Generate with custom options
svg2 = generate_svg_diagram(
    zx_diagram2,
    width=900,
    height=500,
    spider_radius=30,
    show_phases=True,
    show_labels=True
)

save_svg_to_file(svg2, "t_circuit_zx.svg")

print(f"Generated visualization with {len(zx_diagram2.spiders)} spiders")

# Example 3: Accessing SVG as string for embedding
print("\n=== Example 3: SVG String ===")
print("First 200 characters of SVG:")
print(svg[:200])
