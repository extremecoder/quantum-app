import pytest
import logging
import math
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- QASM Content ---
# Using the alternative strategy: Embed QASM content directly
# This avoids potential issues with relative path assumptions during test execution.
QASM_FILENAME = "unknown_circuit_2q.qasm" # Hypothetical original filename for context
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
    """Loads the QuantumCircuit from the QASM string fixture."""
    try:
        circuit = QuantumCircuit.from_qasm_str(qasm_content)
        logging.info(f"Successfully loaded QuantumCircuit from QASM string.")
        logging.info(f"Circuit details: {circuit.num_qubits} qubits, {circuit.num_clbits} classical bits, depth {circuit.depth()}")
        return circuit
    except Exception as e:
        logging.error(f"Failed to load QuantumCircuit from QASM string: {e}")
        pytest.fail(f"Failed to load QuantumCircuit: {e}")

@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Provides an AerSimulator instance."""
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure of the loaded quantum circuit.
    """
    logging.info("Running structural tests on the quantum circuit...")
    assert quantum_circuit is not None, "Quantum circuit object should not be None"
    assert quantum_circuit.num_qubits == 2, f"Expected 2 qubits, but got {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == 2, f"Expected 2 classical bits, but got {quantum_circuit.num_clbits}"

    ops_counts = quantum_circuit.count_ops()
    logging.info(f"Circuit operations: {ops_counts}")
    assert 'h' in ops_counts, "Circuit should contain Hadamard (h) gates"
    assert ops_counts['h'] == 1, "Expected 1 Hadamard gate"
    assert 'cx' in ops_counts, "Circuit should contain CNOT (cx) gates"
    assert ops_counts['cx'] == 1, "Expected 1 CNOT gate"
    assert 'measure' in ops_counts, "Circuit should contain measurement operations"
    assert ops_counts['measure'] == 2, "Expected 2 measurement operations"
    logging.info("Structural tests passed.")

def test_circuit_simulation_and_bell_state(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on a simulator and checks the results,
    specifically verifying the expected Bell state outcomes ('00' and '11').
    """
    logging.info("Running simulation tests on the quantum circuit...")
    shots = 1024  # Number of simulation shots

    # Transpile for the simulator
    transpiled_circuit = transpile(quantum_circuit, simulator)

    # Run the simulation
    job = simulator.run(transpiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(transpiled_circuit)

    logging.info(f"Simulation finished. Shots={shots}, Results: {counts}")

    # --- Simulation Assertions ---
    assert counts is not None, "Simulation should produce counts dictionary"
    assert len(counts) > 0, "Counts dictionary should not be empty"

    total_counts = sum(counts.values())
    assert total_counts == shots, f"Total counts ({total_counts}) should equal the number of shots ({shots})"

    # Check format of result keys (bitstrings)
    expected_classical_bits = quantum_circuit.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Result key '{bitstring}' should be a string"
        # Qiskit counts keys are often reversed (c[1]c[0]), check length only
        assert len(bitstring) == expected_classical_bits, \
            f"Bitstring key '{bitstring}' should have length {expected_classical_bits}"

    # --- Algorithm-Specific Assertions (Bell State) ---
    # This circuit creates a Bell state (|00> + |11>) / sqrt(2).
    # We expect only '00' and '11' outcomes (Qiskit measures c[1]c[0]).
    # Note: Qiskit's counts keys map classical registers c[n-1]...c[0].
    # So, measure q[0]->c[0], q[1]->c[1] results in keys 'c[1]c[0]'.
    # The expected states |00> and |11> correspond to keys '00' and '11'.
    allowed_states = {'00', '11'}
    measured_states = set(counts.keys())

    logging.info(f"Expected states: {allowed_states}, Measured states: {measured_states}")
    assert measured_states.issubset(allowed_states), \
        f"Only states {allowed_states} should be measured, but got {measured_states}"
    assert '00' in counts, "Expected '00' state to be present in results"
    assert '11' in counts, "Expected '11' state to be present in results"

    # Optional: Check for roughly equal probability (within tolerance)
    # This is probabilistic, so use approximate comparison
    expected_prob = 0.5
    tolerance = 0.1 # Allow for 10% deviation due to statistical noise
    prob_00 = counts.get('00', 0) / shots
    prob_11 = counts.get('11', 0) / shots
    logging.info(f"Measured probabilities: P(00)={prob_00:.3f}, P(11)={prob_11:.3f}")
    assert math.isclose(prob_00, expected_prob, rel_tol=tolerance*2), \
        f"Probability of '00' ({prob_00:.3f}) is too far from expected {expected_prob:.3f}"
    assert math.isclose(prob_11, expected_prob, rel_tol=tolerance*2), \
        f"Probability of '11' ({prob_11:.3f}) is too far from expected {expected_prob:.3f}"


    logging.info("Simulation and Bell state tests passed.")

# Example of how to run this test file:
# 1. Save the code as a Python file (e.g., test_bell_circuit.py) in your tests directory.
# 2. Ensure qiskit, qiskit-aer, and pytest are installed (`pip install qiskit qiskit-aer pytest`).
# 3. Run pytest from your terminal in the project's root directory: `pytest`
