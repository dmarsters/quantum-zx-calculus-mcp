"""
Layers 2-3: ZX-Calculus Core Implementation

Layer 2 (Structure): Deterministic circuit analysis and optimization selection
Layer 3 (Relational): Circuit composition rules, rewrite sequence planning

Provides deterministic mapping from quantum circuits to ZX-diagrams with
simplification strategies. Zero LLM cost for these layers.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict

from quantum_zx_ologs import (
    QUANTUM_GATE_TAXONOMY,
    SPIDER_FUSION_RULES,
    lookup_gate,
    SpiderType,
    QuantumGate
)


class CircuitFormat(Enum):
    """Supported quantum circuit formats."""
    QASM = "qasm"
    QISKIT = "qiskit"
    QUIPPER = "quipper"
    ZX_DIAGRAM = "zx_diagram"


class OptimizationStrategy(Enum):
    """Strategies for circuit simplification."""
    CLIFFORD_SIMPLIFICATION = "clifford"
    T_COUNT_REDUCTION = "t_count"
    MEASUREMENT_BASED = "mbqc"
    ERROR_CORRECTION = "error_correction"
    FULL_REDUCTION = "full"
    EDUCATIONAL = "educational"


@dataclass
class QuantumCircuit:
    """Parsed quantum circuit."""
    num_qubits: int
    gates: List[Tuple[str, List[int]]]  # (gate_name, [qubit_indices])
    classical_bits: int = 0
    measurements: List[Tuple[int, int]] = None  # (qubit, classical_bit)
    
    def __post_init__(self):
        if self.measurements is None:
            self.measurements = []


@dataclass
class ZXDiagram:
    """ZX-diagram representation of a quantum circuit."""
    spiders: List[Dict]  # List of spider definitions
    wires: List[Tuple[int, int]]  # Connections between spiders
    phases: Dict[int, float]  # Phase values for each spider
    bounding_box: Optional[Tuple[int, int]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "spiders": self.spiders,
            "wires": self.wires,
            "phases": self.phases,
            "bounding_box": self.bounding_box
        }


@dataclass
class CircuitAnalysis:
    """Analysis of circuit properties and optimization potential."""
    total_gates: int
    clifford_gates: int
    t_gates: int
    two_qubit_gates: int
    measurements: int
    entanglement_score: float  # 0-1 estimate of entanglement
    clifford_depth: int
    estimated_t_count: int
    has_measurements: bool
    is_clifford_only: bool
    recommended_strategies: List[OptimizationStrategy]
    potential_optimizations: Dict[str, str]


# ============================================================================
# LAYER 2: CIRCUIT PARSING (Deterministic)
# ============================================================================

def parse_qasm_circuit(qasm_code: str) -> Optional[QuantumCircuit]:
    """
    Parse OpenQASM 2.0 circuit (deterministic, 0 tokens).
    
    Extracts gate sequence and qubit mapping.
    """
    lines = qasm_code.strip().split('\n')
    num_qubits = 0
    gates = []
    
    for line in lines:
        line = line.strip()
        
        # Extract qubit count
        if line.startswith('qreg'):
            match = re.search(r'qreg\s+\w+\[(\d+)\]', line)
            if match:
                num_qubits = max(num_qubits, int(match.group(1)))
        
        # Parse gate operations
        elif line and not line.startswith('//') and not line.startswith('OPENQASM'):
            # Handle parametric gates
            if '(' in line:
                match = re.match(r'(\w+)\([^)]*\)\s+(q\[\d+\](?:,\s*q\[\d+\])*)', line)
            else:
                match = re.match(r'(\w+)\s+(q\[\d+\](?:,\s*q\[\d+\])*)', line)
            
            if match:
                gate_name = match.group(1)
                qubits_str = match.group(2)
                
                # Extract qubit indices
                qubit_indices = [int(m) for m in re.findall(r'\[(\d+)\]', qubits_str)]
                gates.append((gate_name.lower(), qubit_indices))
    
    if num_qubits == 0 and gates:
        num_qubits = max(max(q for _, qs in gates for q in qs), 0) + 1
    
    return QuantumCircuit(num_qubits=num_qubits, gates=gates) if gates else None


# ============================================================================
# LAYER 2: CIRCUIT ANALYSIS (Deterministic)
# ============================================================================

def analyze_circuit(circuit: QuantumCircuit) -> CircuitAnalysis:
    """
    Analyze circuit properties and determine optimization strategy
    (deterministic, 0 tokens).
    """
    clifford_gates = 0
    t_gates = 0
    two_qubit_gates = 0
    measurements = 0
    entangling_ops = 0
    
    for gate_name, qubits in circuit.gates:
        gate_def = lookup_gate(gate_name)
        if not gate_def:
            continue
        
        if gate_def.is_clifford:
            clifford_gates += 1
        if not gate_def.is_clifford and gate_def.is_clifford_t:
            t_gates += 1
        if len(qubits) >= 2:
            two_qubit_gates += 1
            entangling_ops += 1
        if gate_name in ['measure_z', 'measure_x']:
            measurements += 1
    
    total_gates = len(circuit.gates)
    is_clifford_only = t_gates == 0
    
    # Estimate entanglement (simple heuristic)
    entanglement_score = min(1.0, entangling_ops / max(1, total_gates))
    
    # Estimate clifford depth (simplified)
    clifford_depth = total_gates - t_gates
    
    # Estimate T-count (critical for fault tolerance)
    estimated_t_count = t_gates * 4  # Rough estimate
    
    # Recommend optimization strategies
    recommended_strategies = []
    if is_clifford_only:
        recommended_strategies.append(OptimizationStrategy.CLIFFORD_SIMPLIFICATION)
    if t_gates > 0:
        recommended_strategies.append(OptimizationStrategy.T_COUNT_REDUCTION)
    if measurements > 0:
        recommended_strategies.append(OptimizationStrategy.MEASUREMENT_BASED)
    if entanglement_score > 0.5:
        recommended_strategies.append(OptimizationStrategy.ERROR_CORRECTION)
    recommended_strategies.append(OptimizationStrategy.EDUCATIONAL)
    
    # Potential optimizations
    potential_optimizations = {}
    if clifford_gates > 3:
        potential_optimizations["clifford_simplification"] = "Multiple Clifford gates detected - can be simplified"
    if t_gates > 2:
        potential_optimizations["t_count_reduction"] = f"Circuit has {t_gates} T gates - T-count reduction applicable"
    if two_qubit_gates > 3:
        potential_optimizations["entanglement_routing"] = "Multiple entangling gates - routing optimization possible"
    
    return CircuitAnalysis(
        total_gates=total_gates,
        clifford_gates=clifford_gates,
        t_gates=t_gates,
        two_qubit_gates=two_qubit_gates,
        measurements=measurements,
        entanglement_score=entanglement_score,
        clifford_depth=clifford_depth,
        estimated_t_count=estimated_t_count,
        has_measurements=measurements > 0,
        is_clifford_only=is_clifford_only,
        recommended_strategies=recommended_strategies,
        potential_optimizations=potential_optimizations
    )


# ============================================================================
# LAYER 2: ZX-DIAGRAM CONVERSION (Deterministic)
# ============================================================================

def circuit_to_zx_diagram(circuit: QuantumCircuit) -> ZXDiagram:
    """
    Convert quantum circuit to ZX-diagram representation
    (deterministic, 0 tokens).
    
    Maps gates to spiders and builds wire connectivity.
    """
    spiders = []
    wires = []
    phases = {}
    spider_id = 0
    qubit_to_spider: Dict[int, int] = {}
    
    # Create initial state spiders (one per qubit)
    for qubit in range(circuit.num_qubits):
        spiders.append({
            "id": spider_id,
            "type": "Z",
            "inputs": 0,
            "outputs": 1,
            "label": f"q{qubit}_init"
        })
        phases[spider_id] = 0.0
        qubit_to_spider[qubit] = spider_id
        spider_id += 1
    
    # Process gates
    for gate_name, qubits in circuit.gates:
        gate_def = lookup_gate(gate_name)
        if not gate_def:
            continue
        
        spider_type = gate_def.primary_spider.value
        phase = gate_def.phase
        
        if len(qubits) == 1:
            # Single-qubit gate
            input_spider = qubit_to_spider[qubits[0]]
            spiders.append({
                "id": spider_id,
                "type": spider_type,
                "inputs": 1,
                "outputs": 1,
                "label": gate_name
            })
            phases[spider_id] = phase
            wires.append((input_spider, spider_id))
            qubit_to_spider[qubits[0]] = spider_id
            spider_id += 1
            
        elif len(qubits) == 2:
            # Two-qubit gate (CNOT, CZ, SWAP)
            input1 = qubit_to_spider[qubits[0]]
            input2 = qubit_to_spider[qubits[1]]
            
            # Create gate spider
            spiders.append({
                "id": spider_id,
                "type": spider_type,
                "inputs": 2,
                "outputs": 2,
                "label": gate_name
            })
            phases[spider_id] = phase
            wires.append((input1, spider_id))
            wires.append((input2, spider_id))
            
            current_spider = spider_id
            spider_id += 1
            
            # Update qubit mappings
            qubit_to_spider[qubits[0]] = current_spider
            qubit_to_spider[qubits[1]] = current_spider
    
    return ZXDiagram(
        spiders=spiders,
        wires=wires,
        phases=phases,
        bounding_box=(circuit.num_qubits, len(circuit.gates))
    )


# ============================================================================
# LAYER 2: REWRITE STRATEGY SELECTION (Deterministic)
# ============================================================================

def select_simplification_strategy(analysis: CircuitAnalysis, 
                                  desired_outcome: str = "balanced") -> Dict:
    """
    Select ZX-calculus rewrite strategy based on circuit analysis
    (deterministic, 0 tokens).
    """
    strategy = {
        "primary_goal": desired_outcome,
        "rewrite_rules": [],
        "expected_gate_reduction": 0.0,
        "expected_t_reduction": 0,
        "implementation_notes": []
    }
    
    if desired_outcome == "clifford_simplification":
        strategy["rewrite_rules"] = [
            "spider_fusion",
            "phase_cancellation",
            "hadamard_removal"
        ]
        strategy["expected_gate_reduction"] = min(0.3, analysis.clifford_gates / max(1, analysis.total_gates))
        strategy["implementation_notes"] = [
            "Apply fusion rules to connected spiders of same type",
            "Cancel phases that sum to 2π",
            "Remove Hadamard pairs"
        ]
        
    elif desired_outcome == "t_count_reduction":
        strategy["rewrite_rules"] = [
            "phase_teleportation",
            "color_change",
            "local_complementation"
        ]
        strategy["expected_t_reduction"] = int(analysis.estimated_t_count * 0.3)
        strategy["implementation_notes"] = [
            "Identify T gates (π/4 phases)",
            "Apply phase teleportation to reduce T-count",
            "Use color change to reposition expensive operations"
        ]
        
    elif desired_outcome == "measurement_based":
        strategy["rewrite_rules"] = [
            "measurement_spider",
            "byproduct_correction",
            "flow_verification"
        ]
        strategy["implementation_notes"] = [
            "Represent measurements as X-spiders with parameters",
            "Verify flow structure for measurement-based computation",
            "Compute byproduct corrections"
        ]
        
    elif desired_outcome == "educational":
        strategy["rewrite_rules"] = [
            "step_by_step_fusion",
            "visualization_friendly",
            "explanation_generation"
        ]
        strategy["expected_gate_reduction"] = 0.2
        strategy["implementation_notes"] = [
            "Generate step-by-step transformations",
            "Create SVG visualizations at each step",
            "Generate natural language explanations"
        ]
    
    return strategy


# ============================================================================
# LAYER 2-3: CIRCUIT STATISTICS
# ============================================================================

def get_circuit_statistics(circuit: QuantumCircuit) -> Dict:
    """Get comprehensive circuit statistics (deterministic)."""
    analysis = analyze_circuit(circuit)
    return {
        "basic_info": {
            "num_qubits": circuit.num_qubits,
            "num_gates": analysis.total_gates,
            "num_classical_bits": circuit.classical_bits,
            "num_measurements": analysis.measurements
        },
        "gate_composition": {
            "clifford_gates": analysis.clifford_gates,
            "t_gates": analysis.t_gates,
            "two_qubit_gates": analysis.two_qubit_gates,
            "single_qubit_gates": analysis.total_gates - analysis.two_qubit_gates
        },
        "complexity_metrics": {
            "clifford_depth": analysis.clifford_depth,
            "estimated_t_count": analysis.estimated_t_count,
            "entanglement_score": round(analysis.entanglement_score, 3),
            "is_clifford_only": analysis.is_clifford_only
        },
        "optimization_potential": {
            "recommended_strategies": [s.value for s in analysis.recommended_strategies],
            "potential_optimizations": analysis.potential_optimizations
        }
    }
