import pytest
import logging
import os
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers.backend import Backend
from qiskit.result import Result
from qiskit.result.counts import Counts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- QASM Content ---
# Using the embedded string alternative for robustness as requested.
QASM_CONTENT = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""

# --- Fixtures ---

@pytest.fixture(scope="module")
def qasm_content() -> str:
    """Provides the QASM content as a string."""
    return QASM_CONTENT

@pytest.fixture(scope="module")
def quantum_circuit(qasm_content: str) -> QuantumCircuit:
    """Loads the QuantumCircuit from the QASM string."""
    try:
        circuit = QuantumCircuit.from_qasm_str(qasm_content)
        logging.info("Successfully loaded QuantumCircuit from QASM string.")
        return circuit
    except Exception as e:
        logging.error(f"Failed to load QuantumCircuit from QASM string: {e}")
        pytest.fail(f"Failed to load QuantumCircuit: {e}")

@pytest.fixture(scope="module")
def simulator() -> Backend:
    """Provides an AerSimulator instance."""
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure of the loaded quantum circuit.
    """
    logging.info("Running test_circuit_structure...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."

    # Check number of qubits and classical bits
    expected_qubits = 2
    expected_clbits = 2
    assert quantum_circuit.num_qubits == expected_qubits, \
        f"Expected {expected_qubits} qubits, but got {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, \
        f"Expected {expected_clbits} classical bits, but got {quantum_circuit.num_clbits}"
    logging.info(f"Circuit has {quantum_circuit.num_qubits} qubits and {quantum_circuit.num_clbits} classical bits.")

    # Check for presence of expected gate types
    gate_counts = quantum_circuit.count_ops()
    logging.info(f"Circuit operations: {gate_counts}")
    assert 'h' in gate_counts, "Hadamard gate 'h' not found in the circuit."
    assert 'cx' in gate_counts, "CNOT gate 'cx' not found in the circuit."
    assert 'measure' in gate_counts, "Measure operation not found in the circuit."
    assert gate_counts.get('h', 0) == 1, f"Expected 1 'h' gate, found {gate_counts.get('h', 0)}"
    assert gate_counts.get('cx', 0) == 1, f"Expected 1 'cx' gate, found {gate_counts.get('cx', 0)}"
    assert gate_counts.get('measure', 0) == 2, f"Expected 2 'measure' operations, found {gate_counts.get('measure', 0)}"

    logging.info("test_circuit_structure PASSED.")

def test_circuit_simulation(quantum_circuit: QuantumCircuit, simulator: Backend):
    """
    Runs the circuit on the simulator and performs basic checks on the results.
    """
    logging.info("Running test_circuit_simulation...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."
    assert simulator is not None, "Simulator fixture failed to load."

    shots = 4096
    logging.info(f"Simulating circuit with {shots} shots using {simulator.name}...")

    # Qiskit Aer recommends transpiling for the backend
    try:
        transpiled_circuit = transpile(quantum_circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        pytest.fail(f"Simulation failed: {e}")

    logging.info(f"Simulation finished. Raw counts: {counts}")

    # Basic assertions on results
    assert counts is not None, "Simulation did not return counts."
    assert isinstance(counts, Counts), f"Expected results to be of type Counts, but got {type(counts)}."
    assert sum(counts.values()) == shots, \
        f"Total counts ({sum(counts.values())}) do not match the number of shots ({shots})."

    # Check format of result keys (bitstrings)
    num_clbits = quantum_circuit.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' is not a string."
        assert len(bitstring) == num_clbits, \
            f"Bitstring '{bitstring}' has length {len(bitstring)}, expected {num_clbits}."
        assert all(c in '01' for c in bitstring), \
            f"Bitstring '{bitstring}' contains invalid characters."

    logging.info("test_circuit_simulation PASSED (basic checks).")

def test_bell_state_outcomes(quantum_circuit: QuantumCircuit, simulator: Backend):
    """
    Tests the expected outcomes for a Bell state preparation circuit (|00> + |11>)/sqrt(2).
    """
    logging.info("Running test_bell_state_outcomes...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."
    assert simulator is not None, "Simulator fixture failed to load."

    shots = 4096
    logging.info(f"Simulating Bell state circuit with {shots} shots using {simulator.name}...")

    # Transpile and run
    try:
        transpiled_circuit = transpile(quantum_circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(transpiled_circuit)
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        pytest.fail(f"Simulation failed: {e}")

    logging.info(f"Simulation finished. Raw counts for Bell state test: {counts}")
    assert sum(counts.values()) == shots, "Total counts mismatch."

    # Expected outcomes for |00> + |11> state are '00' and '11'
    # Ideally, '01' and '10' should have zero probability.
    # Allow for some noise/statistical fluctuation in the simulator.
    allowed_error_fraction = 0.05 # Allow up to 5% of shots for unexpected states

    count_00 = counts.get('00', 0)
    count_11 = counts.get('11', 0)
    count_01 = counts.get('01', 0)
    count_10 = counts.get('10', 0)

    logging.info(f"Counts: 00={count_00}, 11={count_11}, 01={count_01}, 10={count_10}")

    # Assert that the primary Bell states are the dominant outcomes
    assert count_00 > 0, "Expected state '00' not found in results."
    assert count_11 > 0, "Expected state '11' not found in results."
    assert count_00 + count_11 > shots * (1 - allowed_error_fraction * 2), \
        f"Combined counts for '00' and '11' ({count_00 + count_11}) are unexpectedly low."

    # Assert that the unexpected states have low probability
    assert count_01 <= shots * allowed_error_fraction, \
        f"Count for unexpected state '01' ({count_01}) exceeds tolerance ({shots * allowed_error_fraction})."
    assert count_10 <= shots * allowed_error_fraction, \
        f"Count for unexpected state '10' ({count_10}) exceeds tolerance ({shots * allowed_error_fraction})."

    # Check for other unexpected keys (shouldn't happen with 2 classical bits)
    unexpected_keys = set(counts.keys()) - {'00', '11', '01', '10'}
    assert not unexpected_keys, f"Found unexpected result keys: {unexpected_keys}"

    logging.info("test_bell_state_outcomes PASSED.")

# Example of how to run this file:
# Save the code as e.g., test_bell_circuit.py in a tests directory
# Ensure qiskit and qiskit-aer are installed: pip install qiskit qiskit-aer pytest
# Run pytest from the terminal in the directory containing the tests directory: pytest
