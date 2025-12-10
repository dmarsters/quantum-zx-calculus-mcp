"""
Layer 1: Foundation - Quantum Gate Taxonomy and ZX-Calculus Primitives

This layer defines the deterministic taxonomy of:
- Quantum gates and their ZX-diagram equivalents
- Spider types and phase parameters
- Measurement operators
- Basic composition rules

All lookups are deterministic with zero LLM cost.
Based on categorical quantum mechanics (Coecke & Duncan, 2008+).
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SpiderType(Enum):
    """Spider types in ZX-calculus."""
    Z_SPIDER = "Z"  # Green spider (computational basis)
    X_SPIDER = "X"  # Red spider (Hadamard basis)
    H_BOX = "H"     # Yellow box (Hadamard gate)


class BasisType(Enum):
    """Measurement bases."""
    COMPUTATIONAL = "computational"  # |0⟩, |1⟩ basis
    HADAMARD = "hadamard"            # |+⟩, |-⟩ basis


@dataclass
class QuantumGate:
    """Definition of a quantum gate and its ZX-diagram representation."""
    name: str
    qiskit_name: str
    matrix_symbol: str
    primary_spider: SpiderType
    phase: float  # In multiples of π
    description: str
    is_clifford: bool
    is_clifford_t: bool
    qasm_format: str


@dataclass
class ZXSpider:
    """A spider in the ZX-diagram."""
    spider_type: SpiderType
    phase: float  # In multiples of π
    inputs: int
    outputs: int
    label: Optional[str] = None


# ============================================================================
# LAYER 1: GATE TAXONOMY
# ============================================================================

QUANTUM_GATE_TAXONOMY: Dict[str, QuantumGate] = {
    # Identity and phase gates
    "i": QuantumGate(
        name="Identity",
        qiskit_name="id",
        matrix_symbol="I",
        primary_spider=SpiderType.Z_SPIDER,
        phase=0.0,
        description="Identity gate - no operation",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="id"
    ),
    "z": QuantumGate(
        name="Pauli Z",
        qiskit_name="z",
        matrix_symbol="Z",
        primary_spider=SpiderType.Z_SPIDER,
        phase=1.0,
        description="Pauli Z gate - phase flip",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="z"
    ),
    "x": QuantumGate(
        name="Pauli X",
        qiskit_name="x",
        matrix_symbol="X",
        primary_spider=SpiderType.X_SPIDER,
        phase=1.0,
        description="Pauli X gate - bit flip",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="x"
    ),
    "y": QuantumGate(
        name="Pauli Y",
        qiskit_name="y",
        matrix_symbol="Y",
        primary_spider=SpiderType.Z_SPIDER,
        phase=1.0,
        description="Pauli Y gate - combined X and Z",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="y"
    ),
    
    # Hadamard and phase gates
    "h": QuantumGate(
        name="Hadamard",
        qiskit_name="h",
        matrix_symbol="H",
        primary_spider=SpiderType.H_BOX,
        phase=0.0,
        description="Hadamard gate - basis change",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="h"
    ),
    "s": QuantumGate(
        name="S (Phase) Gate",
        qiskit_name="s",
        matrix_symbol="S",
        primary_spider=SpiderType.Z_SPIDER,
        phase=0.5,
        description="S gate - π/2 phase",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="s"
    ),
    "sdg": QuantumGate(
        name="S† (Inverse Phase)",
        qiskit_name="sdg",
        matrix_symbol="S†",
        primary_spider=SpiderType.Z_SPIDER,
        phase=-0.5,
        description="Inverse S gate - -π/2 phase",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="sdg"
    ),
    "t": QuantumGate(
        name="T Gate",
        qiskit_name="t",
        matrix_symbol="T",
        primary_spider=SpiderType.Z_SPIDER,
        phase=0.25,
        description="T gate - π/4 phase",
        is_clifford=False,
        is_clifford_t=True,
        qasm_format="t"
    ),
    "tdg": QuantumGate(
        name="T† (Inverse T)",
        qiskit_name="tdg",
        matrix_symbol="T†",
        primary_spider=SpiderType.Z_SPIDER,
        phase=-0.25,
        description="Inverse T gate - -π/4 phase",
        is_clifford=False,
        is_clifford_t=True,
        qasm_format="tdg"
    ),
    
    # Two-qubit gates
    "cx": QuantumGate(
        name="CNOT",
        qiskit_name="cx",
        matrix_symbol="CX",
        primary_spider=SpiderType.Z_SPIDER,
        phase=0.0,
        description="Controlled-NOT - fundamental entangling gate",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="cx"
    ),
    "cz": QuantumGate(
        name="Controlled-Z",
        qiskit_name="cz",
        matrix_symbol="CZ",
        primary_spider=SpiderType.Z_SPIDER,
        phase=0.0,
        description="Controlled phase - alternative entangling gate",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="cz"
    ),
    "swap": QuantumGate(
        name="SWAP",
        qiskit_name="swap",
        matrix_symbol="SWAP",
        primary_spider=SpiderType.Z_SPIDER,
        phase=0.0,
        description="Qubit swap - exchanges two qubits",
        is_clifford=True,
        is_clifford_t=True,
        qasm_format="swap"
    ),
}


# ============================================================================
# LAYER 1: SPIDER REWRITE RULES
# ============================================================================

SPIDER_FUSION_RULES: Dict[str, dict] = {
    "same_type_fusion": {
        "description": "Two spiders of same type connected by single wire fuse",
        "rule": "Z + Z → Z (phases add mod 2π)",
        "example": "Z(π/4) + Z(π/4) → Z(π/2)",
        "preserves_semantics": True
    },
    "phase_cancellation": {
        "description": "Phases that sum to 2π disappear",
        "rule": "Z(π) + Z(π) → Z(0) → identity",
        "example": "X(π) · X(π) = identity",
        "preserves_semantics": True
    },
    "hadamard_removal": {
        "description": "H-box cancels out if adjacent to same basis spider",
        "rule": "H · H = identity",
        "example": "Hadamard conjugation simplifies",
        "preserves_semantics": True
    },
    "spider_bialgebra": {
        "description": "Frobenius algebra structure of spiders",
        "rule": "Multi-leg spiders compose via bialgebra relations",
        "example": "Entanglement preservation in measurements",
        "preserves_semantics": True
    },
    "color_change": {
        "description": "Color change around H-box",
        "rule": "H · Z · H = X (changes basis)",
        "example": "Z → X when conjugated by Hadamard",
        "preserves_semantics": True
    }
}


# ============================================================================
# LAYER 1: MEASUREMENT OPERATORS
# ============================================================================

MEASUREMENT_OPERATORS: Dict[str, dict] = {
    "measure_z": {
        "name": "Z-basis measurement",
        "basis": BasisType.COMPUTATIONAL,
        "represented_as": "X-spider with phase parameter",
        "description": "Measurement in computational basis |0⟩, |1⟩",
        "classical_outcome_representation": "2-bit parameter on X-spider"
    },
    "measure_x": {
        "name": "X-basis measurement",
        "basis": BasisType.HADAMARD,
        "represented_as": "Z-spider with phase parameter",
        "description": "Measurement in Hadamard basis |+⟩, |-⟩",
        "classical_outcome_representation": "2-bit parameter on Z-spider"
    },
    "measure_y": {
        "name": "Y-basis measurement",
        "basis": BasisType.HADAMARD,  # Rotated Hadamard basis
        "represented_as": "Z-spider preceded by Hadamard",
        "description": "Measurement in Y basis",
        "classical_outcome_representation": "Combination of H + Z measurement"
    }
}


# ============================================================================
# LAYER 1: QUANTUM STATES
# ============================================================================

QUANTUM_STATES: Dict[str, dict] = {
    "ket_0": {
        "notation": "|0⟩",
        "zx_representation": "Green spider with 0 legs (state)",
        "phase": 0.0,
        "description": "Computational basis state |0⟩"
    },
    "ket_1": {
        "notation": "|1⟩",
        "zx_representation": "Green spider with π phase, 0 legs",
        "phase": 1.0,
        "description": "Computational basis state |1⟩"
    },
    "ket_plus": {
        "notation": "|+⟩",
        "zx_representation": "Red spider with 0 legs (state)",
        "phase": 0.0,
        "description": "Hadamard basis state |+⟩ = (|0⟩ + |1⟩)/√2"
    },
    "ket_minus": {
        "notation": "|-⟩",
        "zx_representation": "Red spider with π phase, 0 legs",
        "phase": 1.0,
        "description": "Hadamard basis state |-⟩ = (|0⟩ - |1⟩)/√2"
    },
    "bell_pair": {
        "notation": "|Φ+⟩",
        "zx_representation": "Connected Z and X spiders with cup",
        "phase": 0.0,
        "description": "Maximally entangled Bell state (|00⟩ + |11⟩)/√2"
    }
}


# ============================================================================
# LAYER 1: CLIFFORD VS CLIFFORD+T CLASSIFICATION
# ============================================================================

def get_clifford_gates() -> List[str]:
    """Get all Clifford gates (stabilizer codes)."""
    return [
        name for name, gate in QUANTUM_GATE_TAXONOMY.items()
        if gate.is_clifford
    ]


def get_clifford_t_gates() -> List[str]:
    """Get all Clifford+T gates (universal quantum computation)."""
    return [
        name for name, gate in QUANTUM_GATE_TAXONOMY.items()
        if gate.is_clifford_t
    ]


def get_t_gates() -> List[str]:
    """Get non-Clifford T gates (critical resource for fault tolerance)."""
    return [
        name for name, gate in QUANTUM_GATE_TAXONOMY.items()
        if not gate.is_clifford and gate.is_clifford_t
    ]


# ============================================================================
# LOOKUP FUNCTIONS (Layer 1 API)
# ============================================================================

def lookup_gate(gate_name: str) -> Optional[QuantumGate]:
    """Look up gate definition by name (deterministic, 0 tokens)."""
    return QUANTUM_GATE_TAXONOMY.get(gate_name.lower())


def lookup_gate_zx_representation(gate_name: str) -> Optional[Dict]:
    """Get ZX-diagram representation of a gate."""
    gate = lookup_gate(gate_name)
    if not gate:
        return None
    return {
        "gate_name": gate.name,
        "primary_spider": gate.primary_spider.value,
        "phase": gate.phase,
        "phase_degrees": gate.phase * 180,  # Convert to degrees for readability
        "description": gate.description,
        "is_clifford": gate.is_clifford,
        "zx_representation": f"{gate.primary_spider.value}-spider with phase {gate.phase}π"
    }


def lookup_rewrite_rule(rule_name: str) -> Optional[Dict]:
    """Look up ZX-calculus rewrite rule."""
    return SPIDER_FUSION_RULES.get(rule_name)


def lookup_measurement(measurement_type: str) -> Optional[Dict]:
    """Look up measurement operator definition."""
    return MEASUREMENT_OPERATORS.get(measurement_type)


def get_all_gates() -> List[str]:
    """Get list of all supported gates."""
    return list(QUANTUM_GATE_TAXONOMY.keys())


def get_gate_count_statistics() -> Dict[str, int]:
    """Get statistics on gate coverage."""
    gates = QUANTUM_GATE_TAXONOMY.values()
    return {
        "total_gates": len(QUANTUM_GATE_TAXONOMY),
        "clifford_gates": sum(1 for g in gates if g.is_clifford),
        "clifford_t_gates": sum(1 for g in gates if g.is_clifford_t),
        "non_clifford_gates": sum(1 for g in gates if not g.is_clifford),
        "single_qubit_gates": sum(1 for g in gates if "cx" not in g.name.lower()),
        "two_qubit_gates": sum(1 for g in gates if "cx" in g.name.lower() or "cz" in g.name.lower() or "swap" in g.name.lower()),
    }
