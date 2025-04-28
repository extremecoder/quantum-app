import logging
import pytest
import math
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- QASM Content ---
# Using embedded QASM string as the filename/exact path is unknown.
# This ensures the test is self-contained and runnable.
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
def simulator() -> AerSimulator:
    """Provides an AerSimulator instance."""
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure of the loaded quantum circuit.
    """
    logging.info("Running test_circuit_structure...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."

    # Verify qubit and classical bit counts
    expected_qubits = 2
    expected_clbits = 2
    assert quantum_circuit.num_qubits == expected_qubits, \
        f"Expected {expected_qubits} qubits, but found {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, \
        f"Expected {expected_clbits} classical bits, but found {quantum_circuit.num_clbits}"
    logging.info(f"Circuit has {quantum_circuit.num_qubits} qubits and {quantum_circuit.num_clbits} classical bits.")

    # Verify presence of expected operations
    op_names = [op.name for op in quantum_circuit.data]
    assert "h" in op_names, "Hadamard gate (h) not found in the circuit."
    assert "cx" in op_names, "CNOT gate (cx) not found in the circuit."
    assert "measure" in op_names, "Measure operation not found in the circuit."
    logging.info(f"Found expected operations (h, cx, measure) in the circuit: {op_names}")

    # Verify measurement mapping (optional but good)
    measure_ops = [op for op in quantum_circuit.data if op.name == "measure"]
    assert len(measure_ops) == expected_clbits, f"Expected {expected_clbits} measure operations, found {len(measure_ops)}"
    # Check if qubit 0 is measured to clbit 0 and qubit 1 to clbit 1
    # Note: Qiskit's internal representation might differ slightly, but we can check the qubits/clbits involved
    measured_qubits = {quantum_circuit.find_bit(op.qubits[0]).index for op in measure_ops}
    measured_clbits = {quantum_circuit.find_bit(op.clbits[0]).index for op in measure_ops}
    assert measured_qubits == {0, 1}, f"Expected qubits 0 and 1 to be measured, but got {measured_qubits}"
    assert measured_clbits == {0, 1}, f"Expected classical bits 0 and 1 to receive measurements, but got {measured_clbits}"


def test_circuit_simulation_bell_state(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on the AerSimulator and checks the distribution
    for the expected Bell state (|00> + |11>)/sqrt(2).
    """
    logging.info("Running test_circuit_simulation_bell_state...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."
    assert simulator is not None, "Simulator fixture failed to load."

    shots = 4096
    logging.info(f"Simulating circuit with {shots} shots.")

    # Transpile for the simulator for potential optimization
    try:
        transpiled_circuit = transpile(quantum_circuit, simulator)
    except Exception as e:
        logging.error(f"Failed to transpile circuit: {e}")
        pytest.fail(f"Failed to transpile circuit: {e}")

    # Run the simulation
    try:
        result = simulator.run(transpiled_circuit, shots=shots).result()
        counts = result.get_counts(transpiled_circuit)
    except Exception as e:
        logging.error(f"Circuit simulation failed: {e}")
        pytest.fail(f"Circuit simulation failed: {e}")

    logging.info(f"Simulation finished. Counts: {counts}")

    # --- Assertions on Simulation Results ---
    assert counts is not None, "Simulation result's counts are None."
    assert isinstance(counts, dict), f"Counts should be a dictionary, but got {type(counts)}."

    # Check total counts
    total_counts = sum(counts.values())
    assert total_counts == shots, f"Expected total counts {shots}, but got {total_counts}"

    # Check format of result keys (bitstrings)
    num_clbits = quantum_circuit.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Result key {bitstring} is not a string."
        assert len(bitstring) == num_clbits, \
            f"Expected result keys of length {num_clbits}, but found '{bitstring}' with length {len(bitstring)}."
        assert all(c in '01' for c in bitstring), \
            f"Result key '{bitstring}' contains characters other than '0' or '1'."

    # --- Assertions specific to the Bell state circuit ---
    # This circuit creates the Bell state (|00> + |11>)/sqrt(2).
    # Measurements should yield '00' or '11' with roughly equal probability.
    # Ideally, '01' and '10' should not appear in an ideal simulation.
    expected_states = {'00', '11'}
    observed_states = set(counts.keys())

    assert observed_states.issubset(expected_states), \
        f"Observed states {observed_states} contain unexpected outcomes. Only {expected_states} are expected."

    # Check if both expected states are present (highly likely for enough shots)
    if shots > 10: # Avoid failing for very few shots
      assert '00' in observed_states, "Expected state '00' not found in results."
      assert '11' in observed_states, "Expected state '11' not found in results."

    # Check if probabilities are roughly equal (allow some tolerance)
    # Ideal probability is 0.5 for each.
    tolerance = 0.1 # Allow 10% deviation from the ideal 0.5
    prob_00 = counts.get('00', 0) / shots
    prob_11 = counts.get('11', 0) / shots

    logging.info(f"Observed probabilities: P('00')={prob_00:.4f}, P('11')={prob_11:.4f}")

    assert math.isclose(prob_00, 0.5, abs_tol=tolerance), \
        f"Probability of '00' ({prob_00:.4f}) is too far from the expected 0.5 (tolerance {tolerance})."
    assert math.isclose(prob_11, 0.5, abs_tol=tolerance), \
        f"Probability of '11' ({prob_11:.4f}) is too far from the expected 0.5 (tolerance {tolerance})."

    # Explicitly check that disallowed states have zero counts in this ideal simulation
    assert counts.get('01', 0) == 0, "State '01' should not be present in an ideal simulation."
    assert counts.get('10', 0) == 0, "State '10' should not be present in an ideal simulation."

    logging.info("Simulation results match the expected Bell state distribution.")

