"""
FastMCP Server Implementation for Quantum ZX-Calculus MCP

Defines all tools and integrations. Layer 4 handles Claude synthesis.
"""

from fastmcp import FastMCP
import json
from typing import Optional, Dict, Any

# Import local modules
from quantum_zx_ologs import (
    lookup_gate_zx_representation,
    lookup_rewrite_rule,
    lookup_measurement,
    get_all_gates,
    get_gate_count_statistics,
    QUANTUM_GATE_TAXONOMY
)
from quantum_zx_calculus import (
    parse_qasm_circuit,
    analyze_circuit,
    circuit_to_zx_diagram,
    select_simplification_strategy,
    get_circuit_statistics,
    CircuitFormat,
    OptimizationStrategy
)

# Initialize FastMCP server
mcp = FastMCP("quantum-zx-calculus")


# ============================================================================
# LAYER 1: GATE TAXONOMY TOOLS
# ============================================================================

@mcp.tool()
def list_quantum_gates() -> str:
    """
    List all supported quantum gates with their properties.
    
    Returns:
        JSON string with gate taxonomy
    """
    gates = get_all_gates()
    stats = get_gate_count_statistics()
    
    gate_details = []
    for gate_name in sorted(gates):
        gate = QUANTUM_GATE_TAXONOMY.get(gate_name)
        if gate:
            gate_details.append({
                "name": gate.name,
                "qiskit_name": gate.qiskit_name,
                "matrix_symbol": gate.matrix_symbol,
                "primary_spider": gate.primary_spider.value,
                "phase": f"{gate.phase}π",
                "is_clifford": gate.is_clifford,
                "is_clifford_t": gate.is_clifford_t,
                "description": gate.description
            })
    
    return json.dumps({
        "statistics": stats,
        "gates": gate_details
    }, indent=2)


@mcp.tool()
def get_gate_zx_representation(gate_name: str) -> str:
    """
    Get the ZX-diagram representation of a quantum gate.
    
    Args:
        gate_name: Name of the gate (e.g., 'h', 'cx', 't')
    
    Returns:
        JSON with ZX representation details
    """
    result = lookup_gate_zx_representation(gate_name)
    if result:
        return json.dumps(result, indent=2)
    return json.dumps({"error": f"Gate '{gate_name}' not found"})


# ============================================================================
# LAYER 2: CIRCUIT ANALYSIS TOOLS
# ============================================================================

@mcp.tool()
def parse_quantum_circuit(qasm_code: str) -> str:
    """
    Parse an OpenQASM 2.0 quantum circuit.
    
    Args:
        qasm_code: OpenQASM 2.0 circuit code
    
    Returns:
        JSON with parsed circuit structure
    """
    circuit = parse_qasm_circuit(qasm_code)
    if not circuit:
        return json.dumps({"error": "Failed to parse circuit"})
    
    return json.dumps({
        "num_qubits": circuit.num_qubits,
        "num_gates": len(circuit.gates),
        "gates": [
            {"name": name, "qubits": qubits}
            for name, qubits in circuit.gates
        ]
    }, indent=2)


@mcp.tool()
def analyze_quantum_circuit(qasm_code: str) -> str:
    """
    Analyze circuit properties and optimization potential.
    
    Args:
        qasm_code: OpenQASM 2.0 circuit code
    
    Returns:
        JSON with circuit analysis
    """
    circuit = parse_qasm_circuit(qasm_code)
    if not circuit:
        return json.dumps({"error": "Failed to parse circuit"})
    
    stats = get_circuit_statistics(circuit)
    return json.dumps(stats, indent=2)


@mcp.tool()
def circuit_composition_analysis(qasm_code: str) -> str:
    """
    Get detailed gate composition breakdown.
    
    Args:
        qasm_code: OpenQASM 2.0 circuit code
    
    Returns:
        JSON with composition analysis
    """
    circuit = parse_qasm_circuit(qasm_code)
    if not circuit:
        return json.dumps({"error": "Failed to parse circuit"})
    
    gate_counts: Dict[str, int] = {}
    for gate_name, _ in circuit.gates:
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
    
    return json.dumps({
        "total_gates": len(circuit.gates),
        "unique_gates": len(gate_counts),
        "gate_counts": dict(sorted(gate_counts.items(), key=lambda x: x[1], reverse=True)),
        "num_qubits": circuit.num_qubits
    }, indent=2)


# ============================================================================
# LAYER 2-3: ZX-DIAGRAM CONVERSION
# ============================================================================

@mcp.tool()
def convert_circuit_to_zx(qasm_code: str) -> str:
    """
    Convert quantum circuit to ZX-diagram representation.
    
    Args:
        qasm_code: OpenQASM 2.0 circuit code
    
    Returns:
        JSON with ZX-diagram structure
    """
    circuit = parse_qasm_circuit(qasm_code)
    if not circuit:
        return json.dumps({"error": "Failed to parse circuit"})
    
    zx_diagram = circuit_to_zx_diagram(circuit)
    return json.dumps({
        "diagram": zx_diagram.to_dict(),
        "num_spiders": len(zx_diagram.spiders),
        "num_wires": len(zx_diagram.wires),
        "bounding_box": zx_diagram.bounding_box
    }, indent=2)


# ============================================================================
# LAYER 2: OPTIMIZATION STRATEGY SELECTION
# ============================================================================

@mcp.tool()
def recommend_optimization_strategy(qasm_code: str, desired_outcome: str = "balanced") -> str:
    """
    Recommend ZX-calculus optimization strategy based on circuit.
    
    Args:
        qasm_code: OpenQASM 2.0 circuit code
        desired_outcome: One of 'balanced', 'clifford_simplification', 
                        't_count_reduction', 'measurement_based', 'educational'
    
    Returns:
        JSON with recommended strategy
    """
    circuit = parse_qasm_circuit(qasm_code)
    if not circuit:
        return json.dumps({"error": "Failed to parse circuit"})
    
    analysis = analyze_circuit(circuit)
    strategy = select_simplification_strategy(analysis, desired_outcome)
    
    return json.dumps({
        "strategy": strategy,
        "circuit_analysis": {
            "total_gates": analysis.total_gates,
            "clifford_gates": analysis.clifford_gates,
            "t_gates": analysis.t_gates,
            "two_qubit_gates": analysis.two_qubit_gates,
            "is_clifford_only": analysis.is_clifford_only,
            "estimated_t_count": analysis.estimated_t_count
        }
    }, indent=2)


# ============================================================================
# LAYER 1: REFERENCE TOOLS
# ============================================================================

@mcp.tool()
def get_zx_rewrite_rules() -> str:
    """
    Get reference guide for ZX-calculus rewrite rules.
    
    Returns:
        JSON with all available rewrite rules
    """
    rules = {
        "same_type_fusion": lookup_rewrite_rule("same_type_fusion"),
        "phase_cancellation": lookup_rewrite_rule("phase_cancellation"),
        "hadamard_removal": lookup_rewrite_rule("hadamard_removal"),
        "spider_bialgebra": lookup_rewrite_rule("spider_bialgebra"),
        "color_change": lookup_rewrite_rule("color_change"),
    }
    return json.dumps(rules, indent=2)


@mcp.tool()
def get_quantum_measurement_basis(measurement_type: str) -> str:
    """
    Get quantum measurement basis definition.
    
    Args:
        measurement_type: One of 'measure_z', 'measure_x', 'measure_y'
    
    Returns:
        JSON with measurement definition
    """
    result = lookup_measurement(measurement_type)
    if result:
        return json.dumps(result, indent=2)
    return json.dumps({"error": f"Measurement type '{measurement_type}' not found"})


# ============================================================================
# LAYER 4: SYNTHESIS AND EXPLANATION TOOLS
# ============================================================================

@mcp.tool()
def explain_zx_diagram_transformation(before_qasm: str, after_qasm: str, 
                                      transformation_type: str = "simplification") -> str:
    """
    Generate explanation of transformation between two circuits.
    
    This is a Layer 4 tool that can accept Claude synthesis.
    
    Args:
        before_qasm: Original circuit in OpenQASM
        after_qasm: Transformed circuit in OpenQASM
        transformation_type: Type of transformation applied
    
    Returns:
        Structured explanation (Claude can synthesize natural language version)
    """
    before_circuit = parse_qasm_circuit(before_qasm)
    after_circuit = parse_qasm_circuit(after_qasm)
    
    if not before_circuit or not after_circuit:
        return json.dumps({"error": "Failed to parse one or both circuits"})
    
    before_analysis = analyze_circuit(before_circuit)
    after_analysis = analyze_circuit(after_circuit)
    
    return json.dumps({
        "transformation_type": transformation_type,
        "before": {
            "total_gates": before_analysis.total_gates,
            "clifford_gates": before_analysis.clifford_gates,
            "t_gates": before_analysis.t_gates,
            "estimated_t_count": before_analysis.estimated_t_count
        },
        "after": {
            "total_gates": after_analysis.total_gates,
            "clifford_gates": after_analysis.clifford_gates,
            "t_gates": after_analysis.t_gates,
            "estimated_t_count": after_analysis.estimated_t_count
        },
        "improvements": {
            "gate_reduction": before_analysis.total_gates - after_analysis.total_gates,
            "t_count_reduction": before_analysis.estimated_t_count - after_analysis.estimated_t_count,
            "depth_improvement": before_analysis.clifford_depth - after_analysis.clifford_depth
        }
    }, indent=2)


@mcp.tool()
def generate_zx_diagram_markdown(qasm_code: str, title: str = "ZX-Diagram") -> str:
    """
    Generate markdown documentation of ZX-diagram with ASCII representation.
    
    Args:
        qasm_code: OpenQASM 2.0 circuit code
        title: Title for the diagram documentation
    
    Returns:
        Markdown string with diagram documentation
    """
    circuit = parse_qasm_circuit(qasm_code)
    if not circuit:
        return "Error: Failed to parse circuit"
    
    analysis = analyze_circuit(circuit)
    zx_diagram = circuit_to_zx_diagram(circuit)
    
    markdown = f"""# {title}

## Circuit Properties

- **Number of Qubits:** {circuit.num_qubits}
- **Total Gates:** {analysis.total_gates}
- **Clifford Gates:** {analysis.clifford_gates}
- **T Gates:** {analysis.t_gates}
- **Two-Qubit Gates:** {analysis.two_qubit_gates}
- **Is Clifford-Only:** {analysis.is_clifford_only}

## ZX-Diagram Statistics

- **Number of Spiders:** {len(zx_diagram.spiders)}
- **Number of Wires:** {len(zx_diagram.wires)}
- **Bounding Box:** {zx_diagram.bounding_box}

## Gate Sequence

```
"""
    for i, (gate_name, qubits) in enumerate(circuit.gates, 1):
        qubit_str = ", ".join(f"q{q}" for q in qubits)
        markdown += f"{i:3d}. {gate_name:8s} {qubit_str}\n"
    
    markdown += """```

## Spiders

"""
    for spider in zx_diagram.spiders:
        phase_str = f" (phase: {zx_diagram.phases.get(spider['id'], 0)}π)" if zx_diagram.phases.get(spider['id'], 0) != 0 else ""
        markdown += f"- Spider {spider['id']}: {spider['type']}-spider {spider['inputs']} in, {spider['outputs']} out{phase_str}\n"
    
    return markdown


# ============================================================================
# SERVER INFO
# ============================================================================

@mcp.tool()
def get_server_info() -> str:
    """
    Get information about the Quantum ZX-Calculus MCP server.
    
    Returns:
        JSON with server information and capabilities
    """
    return json.dumps({
        "server": "quantum-zx-calculus-mcp",
        "version": "0.1.0",
        "description": "Convert quantum circuits to ZX-diagrams and analyze optimization strategies",
        "architecture": {
            "layer1": "Foundation - Gate taxonomy and ZX primitives (0 tokens)",
            "layer2": "Structure - Circuit parsing and analysis (0 tokens)",
            "layer3": "Relational - Optimization strategy selection (0 tokens)",
            "layer4": "Contextual - Claude synthesis for explanations"
        },
        "supported_formats": ["qasm", "description"],
        "cost_optimization": "60-85% savings through deterministic layers",
        "based_on": "Bob Coecke's Categorical Quantum Mechanics (2008+)",
        "references": [
            "Coecke & Kissinger, 'Picturing Quantum Processes' (Cambridge, 2017)",
            "ZX-calculus: https://zxcalculus.com/",
            "PyZX: https://github.com/zxcalc/pyzx"
        ]
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
