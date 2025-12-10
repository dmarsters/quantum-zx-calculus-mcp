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
# LAYER 1: INTENT CLASSIFICATION (NATURAL LANGUAGE INTERFACE)
# ============================================================================

# Intent patterns for natural language understanding
INTENT_PATTERNS = {
    "gate_query": {
        "keywords": ["what", "about", "property", "tell me", "explain", "gate", "is the"],
        "description": "Questions about specific quantum gates"
    },
    "circuit_analysis": {
        "keywords": ["analyze", "circuit", "count", "clifford", "entangle", "what does", "what is"],
        "description": "Questions about analyzing existing circuits"
    },
    "optimization": {
        "keywords": ["optimize", "simplify", "reduce", "minimize", "shorten", "improve"],
        "description": "Questions about circuit optimization"
    },
    "zx_diagram": {
        "keywords": ["zx", "spider", "diagram", "visualize", "rewrite", "transformation"],
        "description": "Questions about ZX-diagram representation"
    },
    "learning": {
        "keywords": ["teach", "learn", "show", "example", "demonstrate", "how", "bell state", "make"],
        "description": "Educational questions and examples"
    }
}


def classify_intent(user_question: str) -> tuple[str, float]:
    """
    Classify user question into intent categories (Layer 1).
    
    Pure deterministic classification based on keyword matching.
    No LLM involved. 0 tokens.
    
    Args:
        user_question: Natural language question from user
    
    Returns:
        (intent_name, confidence_score)
    """
    question_lower = user_question.lower()
    scores = {}
    
    for intent, data in INTENT_PATTERNS.items():
        score = 0
        for keyword in data["keywords"]:
            if keyword in question_lower:
                score += 1
        scores[intent] = score
    
    # Find highest scoring intent
    best_intent = max(scores, key=scores.get)
    confidence = scores[best_intent] / sum(scores.values()) if sum(scores.values()) > 0 else 0
    
    # Default to circuit_analysis if no clear match
    if confidence < 0.2:
        best_intent = "circuit_analysis"
        confidence = 0.5
    
    return best_intent, confidence


def extract_gate_name(user_question: str) -> Optional[str]:
    """
    Extract gate name from question.
    
    Layer 1: Pattern matching for gate names.
    
    Args:
        user_question: Question that mentions a gate
    
    Returns:
        Normalized gate name or None
    """
    gate_aliases = {
        "hadamard": "h",
        "pauli-x": "x",
        "pauli_x": "x",
        "pauli-y": "y",
        "pauli_y": "y",
        "pauli-z": "z",
        "pauli_z": "z",
        "cnot": "cx",
        "control-not": "cx",
        "control_not": "cx",
        "control-z": "cz",
        "control_z": "cz",
        "t-gate": "t",
        "s-gate": "s",
        "phase gate": "s"
    }
    
    question_lower = user_question.lower()
    
    # Check aliases first
    for alias, standard_name in gate_aliases.items():
        if alias in question_lower:
            return standard_name
    
    # Check standard single-letter gates
    standard_gates = ["h", "x", "y", "z", "s", "t", "cx", "cz", "swap", "rx", "ry", "rz"]
    for gate in standard_gates:
        if gate in question_lower:
            return gate
    
    return None


def format_gate_response(gate_data: Dict[str, Any]) -> str:
    """
    Format gate information for natural language output.
    
    Returns JSON that Claude will synthesize into prose.
    
    Args:
        gate_data: Gate information from get_gate_zx_representation
    
    Returns:
        JSON with response type and synthesis instruction
    """
    return json.dumps({
        "response_type": "gate_info",
        "data": gate_data,
        "synthesis_instruction": (
            "Format this gate information as a clear, educational explanation. "
            "Include: what the gate does, its ZX representation (spider type and phase), "
            "whether it's Clifford, and where it's used. Keep it concise but complete."
        )
    }, indent=2)


def format_circuit_analysis_response(analysis: Dict[str, Any]) -> str:
    """
    Format circuit analysis for natural language output.
    
    Args:
        analysis: Circuit analysis from analyze_quantum_circuit
    
    Returns:
        JSON with response type and synthesis instruction
    """
    return json.dumps({
        "response_type": "circuit_analysis",
        "data": analysis,
        "synthesis_instruction": (
            "Explain this circuit analysis conversationally. Describe: what the circuit does, "
            "its key properties (Clifford gates, T-count, entanglement score), and any obvious "
            "optimization opportunities. Make it accessible to someone learning quantum computing."
        )
    }, indent=2)


def format_optimization_response(strategy: Dict[str, Any]) -> str:
    """
    Format optimization recommendation for natural language output.
    
    Args:
        strategy: Strategy recommendation from recommend_optimization_strategy
    
    Returns:
        JSON with response type and synthesis instruction
    """
    return json.dumps({
        "response_type": "optimization",
        "data": strategy,
        "synthesis_instruction": (
            "Explain the optimization strategy clearly. Describe: the current circuit state, "
            "which ZX-rewrite rules could apply, what the simplified form would look like, "
            "and what's gained (e.g., '50% gate reduction'). Make it motivating—explain why "
            "the optimization matters."
        )
    }, indent=2)


def format_learning_response(circuit_qasm: str, explanation: Dict[str, Any]) -> str:
    """
    Format educational content for natural language output.
    
    Args:
        circuit_qasm: The circuit being explained
        explanation: Analysis of the circuit
    
    Returns:
        JSON with response type and synthesis instruction
    """
    return json.dumps({
        "response_type": "learning",
        "circuit": circuit_qasm,
        "data": explanation,
        "synthesis_instruction": (
            "Explain this quantum circuit pedagogically. Go step-by-step through what each "
            "gate does, showing the evolving quantum state. Use Dirac notation (|0⟩, |1⟩) "
            "and show the final state. Explain why this circuit is interesting or useful."
        )
    }, indent=2)


def format_zx_response(zx_data: Dict[str, Any]) -> str:
    """
    Format ZX-diagram information for natural language output.
    
    Args:
        zx_data: ZX-diagram from convert_circuit_to_zx
    
    Returns:
        JSON with response type and synthesis instruction
    """
    return json.dumps({
        "response_type": "zx_diagram",
        "data": zx_data,
        "synthesis_instruction": (
            "Explain this ZX-diagram representation. Describe the spiders (nodes), "
            "wires (connections), and phases. Explain what each spider type (Z, X, H) "
            "represents and why this representation is useful for quantum circuit optimization."
        )
    }, indent=2)


def route_and_execute(
    intent: str,
    user_question: str,
    circuit_qasm: Optional[str] = None
) -> Dict[str, Any]:
    """
    Route question to appropriate tools and execute (Layers 2-3).
    
    All execution is deterministic - no LLM involved. 0 tokens.
    
    Args:
        intent: Classified intent from classify_intent
        user_question: Original user question
        circuit_qasm: Optional circuit code
    
    Returns:
        Result dictionary with formatted data
    """
    
    if intent == "gate_query":
        gate_name = extract_gate_name(user_question)
        if not gate_name:
            return {
                "error": "Could not identify which gate you're asking about",
                "suggestion": "Try: 'Tell me about the H-gate' or 'What is a CNOT?'"
            }
        
        result = get_gate_zx_representation(gate_name)
        result_dict = json.loads(result) if isinstance(result, str) else result
        return {
            "intent": intent,
            "gate": gate_name,
            "result": result_dict
        }
    
    elif intent == "circuit_analysis":
        if not circuit_qasm:
            return {
                "error": "Please provide a circuit to analyze",
                "example": "Analyze this circuit: h q[0]; cx q[0],q[1]"
            }
        
        analysis = analyze_quantum_circuit(circuit_qasm)
        analysis_dict = json.loads(analysis) if isinstance(analysis, str) else analysis
        
        return {
            "intent": intent,
            "circuit": circuit_qasm,
            "analysis": analysis_dict
        }
    
    elif intent == "optimization":
        if not circuit_qasm:
            return {
                "error": "Please provide a circuit to optimize",
                "example": "Optimize this circuit: h q[0]; t q[0]; h q[0]; t q[0]"
            }
        
        strategy = recommend_optimization_strategy(circuit_qasm, "balanced")
        strategy_dict = json.loads(strategy) if isinstance(strategy, str) else strategy
        
        return {
            "intent": intent,
            "circuit": circuit_qasm,
            "strategy": strategy_dict
        }
    
    elif intent == "zx_diagram":
        if not circuit_qasm:
            return {
                "error": "Please provide a circuit for ZX-diagram conversion",
                "example": "Show me the ZX-diagram for: h q[0]; cx q[0],q[1]"
            }
        
        zx_result = convert_circuit_to_zx(circuit_qasm)
        zx_dict = json.loads(zx_result) if isinstance(zx_result, str) else zx_result
        
        return {
            "intent": intent,
            "circuit": circuit_qasm,
            "zx_diagram": zx_dict
        }
    
    elif intent == "learning":
        # Common examples
        examples = {
            "bell state": "h q[0]; cx q[0],q[1]",
            "superposition": "h q[0]",
            "entanglement": "cx q[0],q[1]",
            "ghz state": "h q[0]; cx q[0],q[1]; cx q[1],q[2]"
        }
        
        circuit_qasm = None
        for key, value in examples.items():
            if key.lower() in user_question.lower():
                circuit_qasm = value
                break
        
        if not circuit_qasm:
            circuit_qasm = "h q[0]; cx q[0],q[1]"  # Default Bell state
        
        analysis = analyze_quantum_circuit(circuit_qasm)
        analysis_dict = json.loads(analysis) if isinstance(analysis, str) else analysis
        
        return {
            "intent": intent,
            "circuit": circuit_qasm,
            "analysis": analysis_dict
        }
    
    return {"error": f"Unknown intent: {intent}"}


# ============================================================================
# LAYER 4: NATURAL LANGUAGE INTERFACE TOOL
# ============================================================================

@mcp.tool()
def ask_about_quantum_circuit(
    user_question: str,
    circuit_qasm: Optional[str] = None
) -> str:
    """
    Natural language interface to quantum ZX-calculus brick.
    
    Ask questions about quantum circuits in plain English instead of
    calling specific tools. This tool intelligently routes to the
    appropriate analysis tools based on your question.
    
    Args:
        user_question: Your question in natural language. Examples:
            - "Tell me about the T-gate"
            - "What's a Bell state?"
            - "Analyze this circuit"
            - "Optimize this circuit"
            - "Show me the ZX-diagram"
        
        circuit_qasm: Optional OpenQASM circuit code to analyze
    
    Returns:
        Structured data formatted for Claude to synthesize into prose
    
    Examples:
        ask_about_quantum_circuit("Tell me about the T-gate")
        ask_about_quantum_circuit("What's a Bell state?")
        ask_about_quantum_circuit("Optimize this circuit", 
                                  circuit_qasm="h q[0]; t q[0]; h q[0]; t q[0]")
        ask_about_quantum_circuit("Show me entanglement examples")
        ask_about_quantum_circuit("Analyze h q[0]; cx q[0],q[1]; t q[1]")
    
    Cost profile:
        Layer 1 (Intent classification): 0 tokens
        Layer 2 (Tool routing): 0 tokens  
        Layer 3 (Execution): 0 tokens (all deterministic)
        Layer 4 (Synthesis): ~50-100 tokens (Claude formats response)
        Total: ~50-100 tokens vs 400-600 tokens for pure LLM approach
        Savings: 75-85%
    """
    
    # Layer 1: Classify intent (0 tokens)
    intent, confidence = classify_intent(user_question)
    
    # Layer 2-3: Route and execute (0 tokens)
    execution_result = route_and_execute(intent, user_question, circuit_qasm)
    
    # Check for errors
    if "error" in execution_result:
        return json.dumps({
            "status": "error",
            "error_message": execution_result["error"],
            "suggestion": execution_result.get("suggestion", execution_result.get("example")),
            "intent_detected": intent,
            "confidence": confidence
        }, indent=2)
    
    # Layer 4: Format for Claude synthesis
    if intent == "gate_query":
        formatted = format_gate_response(execution_result["result"])
    elif intent == "circuit_analysis":
        formatted = format_circuit_analysis_response(execution_result["analysis"])
    elif intent == "optimization":
        formatted = format_optimization_response(execution_result["strategy"])
    elif intent == "zx_diagram":
        formatted = format_zx_response(execution_result["zx_diagram"])
    elif intent == "learning":
        formatted = format_learning_response(execution_result["circuit"], execution_result["analysis"])
    else:
        formatted = json.dumps(execution_result)
    
    return formatted


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
        "version": "0.2.0",
        "description": "Convert quantum circuits to ZX-diagrams and analyze optimization strategies",
        "architecture": {
            "layer1": "Foundation - Gate taxonomy, intent classification (0 tokens)",
            "layer2": "Structure - Circuit parsing, tool routing (0 tokens)",
            "layer3": "Relational - Optimization strategy selection (0 tokens)",
            "layer4": "Contextual - Claude synthesis for explanations (~50-100 tokens)"
        },
        "supported_formats": ["qasm", "description"],
        "natural_language_interface": "ask_about_quantum_circuit tool",
        "cost_optimization": "75-85% savings through deterministic layers",
        "based_on": "Bob Coecke's Categorical Quantum Mechanics (2008+)",
        "tools": {
            "layer1_gates": ["list_quantum_gates", "get_gate_zx_representation", "get_zx_rewrite_rules", "get_quantum_measurement_basis"],
            "layer2_analysis": ["parse_quantum_circuit", "analyze_quantum_circuit", "circuit_composition_analysis", "convert_circuit_to_zx"],
            "layer3_optimization": ["recommend_optimization_strategy"],
            "layer4_synthesis": ["explain_zx_diagram_transformation", "generate_zx_diagram_markdown", "ask_about_quantum_circuit"],
            "meta": ["get_server_info"]
        },
        "references": [
            "Coecke & Kissinger, 'Picturing Quantum Processes' (Cambridge, 2017)",
            "ZX-calculus: https://zxcalculus.com/",
            "PyZX: https://github.com/zxcalc/pyzx"
        ]
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
