"""
Test suite for quantum-zx-calculus-mcp server

Tests Layer 1-3 deterministic operations and Layer 4 integration points.
"""

import pytest
import json
from quantum_zx_ologs import (
    lookup_gate,
    lookup_gate_zx_representation,
    lookup_rewrite_rule,
    lookup_measurement,
    get_all_gates,
    get_gate_count_statistics,
    QUANTUM_GATE_TAXONOMY
)
from quantum_zx_calculus import (
    QuantumCircuit,
    parse_qasm_circuit,
    analyze_circuit,
    circuit_to_zx_diagram,
    select_simplification_strategy,
    get_circuit_statistics,
    OptimizationStrategy
)


class TestLayer1GateTaxonomy:
    """Test Layer 1: Foundation - Gate Taxonomy"""
    
    def test_gate_lookup_exists(self):
        """Test that common gates can be looked up"""
        assert lookup_gate("h") is not None
        assert lookup_gate("cx") is not None
        assert lookup_gate("t") is not None
    
    def test_gate_lookup_case_insensitive(self):
        """Test gate lookup is case-insensitive"""
        assert lookup_gate("H") is not None
        assert lookup_gate("CX") is not None
    
    def test_gate_properties(self):
        """Test gate definitions have correct properties"""
        h_gate = lookup_gate("h")
        assert h_gate.name == "Hadamard"
        assert h_gate.is_clifford == True
        assert h_gate.is_clifford_t == True
    
    def test_t_gate_not_clifford(self):
        """Test T gate is not Clifford"""
        t_gate = lookup_gate("t")
        assert t_gate.is_clifford == False
        assert t_gate.is_clifford_t == True
    
    def test_zx_representation(self):
        """Test ZX representation lookup"""
        result = lookup_gate_zx_representation("h")
        assert result is not None
        assert "primary_spider" in result
        assert "phase" in result
    
    def test_rewrite_rules_exist(self):
        """Test rewrite rules are defined"""
        rule = lookup_rewrite_rule("same_type_fusion")
        assert rule is not None
        assert "description" in rule
    
    def test_measurement_operators(self):
        """Test measurement operators are defined"""
        meas = lookup_measurement("measure_z")
        assert meas is not None
        assert "basis" in meas
    
    def test_gate_statistics(self):
        """Test gate count statistics"""
        stats = get_gate_count_statistics()
        assert stats["total_gates"] > 0
        assert stats["clifford_gates"] > 0


class TestLayer2CircuitParsing:
    """Test Layer 2: Structure - Circuit Parsing"""
    
    def test_parse_simple_circuit(self):
        """Test parsing simple circuit"""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """
        circuit = parse_qasm_circuit(qasm)
        assert circuit is not None
        assert circuit.num_qubits == 2
        assert len(circuit.gates) == 2
    
    def test_parse_gate_names(self):
        """Test gate names are parsed correctly"""
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        h q[0];
        t q[0];
        """
        circuit = parse_qasm_circuit(qasm)
        gate_names = [name for name, _ in circuit.gates]
        assert "h" in gate_names
        assert "t" in gate_names
    
    def test_parse_qubit_indices(self):
        """Test qubit indices are parsed correctly"""
        qasm = """
        OPENQASM 2.0;
        qreg q[3];
        cx q[0], q[1];
        cx q[1], q[2];
        """
        circuit = parse_qasm_circuit(qasm)
        assert circuit.gates[0][1] == [0, 1]
        assert circuit.gates[1][1] == [1, 2]
    
    def test_parse_multi_gate_circuit(self):
        """Test parsing circuit with multiple gates"""
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        h q[1];
        cx q[0], q[1];
        t q[0];
        """
        circuit = parse_qasm_circuit(qasm)
        assert len(circuit.gates) == 4


class TestLayer2CircuitAnalysis:
    """Test Layer 2: Circuit Analysis"""
    
    def test_analyze_clifford_circuit(self):
        """Test analysis of Clifford-only circuit"""
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """
        circuit = parse_qasm_circuit(qasm)
        analysis = analyze_circuit(circuit)
        assert analysis.is_clifford_only == True
        assert analysis.t_gates == 0
    
    def test_analyze_t_gate_circuit(self):
        """Test analysis detects T gates"""
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        t q[0];
        """
        circuit = parse_qasm_circuit(qasm)
        analysis = analyze_circuit(circuit)
        assert analysis.t_gates == 1
        assert analysis.is_clifford_only == False
    
    def test_analyze_gate_counts(self):
        """Test gate counting in analysis"""
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        h q[1];
        cx q[0], q[1];
        """
        circuit = parse_qasm_circuit(qasm)
        analysis = analyze_circuit(circuit)
        assert analysis.total_gates == 3
        assert analysis.two_qubit_gates == 1
    
    def test_circuit_statistics(self):
        """Test comprehensive circuit statistics"""
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        t q[1];
        """
        circuit = parse_qasm_circuit(qasm)
        stats = get_circuit_statistics(circuit)
        assert "basic_info" in stats
        assert "gate_composition" in stats
        assert "complexity_metrics" in stats
        assert stats["basic_info"]["num_qubits"] == 2
        assert stats["basic_info"]["num_gates"] == 3


class TestLayer2ZXConversion:
    """Test Layer 2: ZX-Diagram Conversion"""
    
    def test_circuit_to_zx_simple(self):
        """Test conversion of simple circuit to ZX"""
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        h q[0];
        """
        circuit = parse_qasm_circuit(qasm)
        zx = circuit_to_zx_diagram(circuit)
        assert zx.spiders is not None
        assert len(zx.spiders) > 0
    
    def test_zx_diagram_structure(self):
        """Test ZX diagram has required structure"""
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """
        circuit = parse_qasm_circuit(qasm)
        zx = circuit_to_zx_diagram(circuit)
        assert "spiders" in zx.to_dict()
        assert "wires" in zx.to_dict()
        assert "phases" in zx.to_dict()
    
    def test_zx_spider_creation(self):
        """Test spiders are created for each gate"""
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        h q[0];
        t q[0];
        h q[0];
        """
        circuit = parse_qasm_circuit(qasm)
        zx = circuit_to_zx_diagram(circuit)
        # Should have initial qubit + 3 gate spiders
        assert len(zx.spiders) >= 4


class TestLayer2OptimizationStrategy:
    """Test Layer 2: Optimization Strategy Selection"""
    
    def test_clifford_simplification_strategy(self):
        """Test Clifford simplification strategy selection"""
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        h q[0];
        cx q[0], q[1];
        """
        circuit = parse_qasm_circuit(qasm)
        analysis = analyze_circuit(circuit)
        strategy = select_simplification_strategy(analysis, "clifford_simplification")
        assert "rewrite_rules" in strategy
        assert len(strategy["rewrite_rules"]) > 0
    
    def test_t_count_reduction_strategy(self):
        """Test T-count reduction strategy for circuits with T gates"""
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        t q[0];
        cx q[0], q[1];
        t q[1];
        """
        circuit = parse_qasm_circuit(qasm)
        analysis = analyze_circuit(circuit)
        strategy = select_simplification_strategy(analysis, "t_count_reduction")
        assert "rewrite_rules" in strategy
        assert strategy["expected_t_reduction"] > 0
    
    def test_educational_strategy(self):
        """Test educational visualization strategy"""
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        h q[0];
        """
        circuit = parse_qasm_circuit(qasm)
        analysis = analyze_circuit(circuit)
        strategy = select_simplification_strategy(analysis, "educational")
        assert "implementation_notes" in strategy


class TestLayer3RecommendedStrategies:
    """Test Layer 3: Recommended optimization strategies"""
    
    def test_clifford_only_circuit_recommendations(self):
        """Test that Clifford circuits get simplification recommendation"""
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """
        circuit = parse_qasm_circuit(qasm)
        analysis = analyze_circuit(circuit)
        assert OptimizationStrategy.CLIFFORD_SIMPLIFICATION in analysis.recommended_strategies
    
    def test_t_gate_circuit_recommendations(self):
        """Test that circuits with T gates get T-count recommendation"""
        qasm = """
        OPENQASM 2.0;
        qreg q[1];
        t q[0];
        """
        circuit = parse_qasm_circuit(qasm)
        analysis = analyze_circuit(circuit)
        assert OptimizationStrategy.T_COUNT_REDUCTION in analysis.recommended_strategies


class TestIntegration:
    """Integration tests across layers"""
    
    def test_full_pipeline(self):
        """Test full pipeline from QASM to analysis to strategy"""
        qasm = """
        OPENQASM 2.0;
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        t q[1];
        """
        
        # Parse
        circuit = parse_qasm_circuit(qasm)
        assert circuit is not None
        
        # Analyze
        analysis = analyze_circuit(circuit)
        assert analysis.total_gates == 3
        
        # Convert to ZX
        zx = circuit_to_zx_diagram(circuit)
        assert len(zx.spiders) > 0
        
        # Get strategy
        strategy = select_simplification_strategy(analysis)
        assert "rewrite_rules" in strategy
    
    def test_complex_circuit(self):
        """Test handling of larger circuit"""
        qasm = """
        OPENQASM 2.0;
        qreg q[4];
        h q[0];
        h q[1];
        cx q[0], q[1];
        cx q[1], q[2];
        h q[3];
        t q[2];
        t q[3];
        cx q[2], q[3];
        """
        circuit = parse_qasm_circuit(qasm)
        assert circuit.num_qubits == 4
        assert len(circuit.gates) == 8
        
        analysis = analyze_circuit(circuit)
        stats = get_circuit_statistics(circuit)
        assert stats["basic_info"]["num_qubits"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
