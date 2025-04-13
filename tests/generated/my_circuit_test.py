import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicSimulator  # Basic simulator for comparison if needed
from qiskit_aer import AerSimulator  # Preferred modern simulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Test Configuration ---
SHOTS = 4096  # Number of shots for simulation tests

# QASM content embedded as a string (Chosen over file loading for robustness in this context)
# This corresponds to the circuit provided in the prompt.
BELL_STATE_QASM_CONTENT = """OPENQASM 2.0;
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
    logging.info("Providing QASM content fixture.")
    return BELL_STATE_QASM_CONTENT

@pytest.fixture(scope="module")
def quantum_circuit(qasm_content: str) -> QuantumCircuit:
    """
    Loads the QuantumCircuit from the QASM string fixture.
    Handles potential loading errors.
    """
    try:
        circuit = QuantumCircuit.from_qasm_str(qasm_content)
        logging.info("Successfully loaded QuantumCircuit from QASM string.")
        logging.info(f"Circuit Name: {circuit.name}, Num Qubits: {circuit.num_qubits}, Num Clbits: {circuit.num_clbits}")
        return circuit
    except Exception as e:
        logging.error(f"Failed to load QuantumCircuit from QASM string: {e}", exc_info=True)
        pytest.fail(f"Failed to load QuantumCircuit from QASM string: {e}")

@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Provides a high-performance AerSimulator instance."""
    logging.info("Providing AerSimulator fixture.")
    # Configure simulator if needed (e.g., noise model, method)
    # sim = AerSimulator(method='statevector') # Example configuration
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure of the loaded quantum circuit.
    Checks qubit/clbit counts and the presence/count of expected operations.
    """
    qc = quantum_circuit
    assert qc is not None, "QuantumCircuit fixture failed to load."
    logging.info(f"Running structural tests on circuit '{qc.name}'")

    # Assert basic properties
    assert qc.num_qubits == 2, f"Expected 2 qubits, but got {qc.num_qubits}"
    assert qc.num_clbits == 2, f"Expected 2 classical bits, but got {qc.num_clbits}"
    assert qc.depth() > 0, "Circuit depth should be greater than 0 for a non-empty circuit."

    # Assert presence and count of expected gate types
    ops = qc.count_ops()
    logging.info(f"Circuit operations: {ops}")
    expected_ops = {"h": 1, "cx": 1, "measure": 2}

    assert "h" in ops, "Hadamard gate (h) not found in the circuit."
    assert ops["h"] == expected_ops["h"], \
        f"Expected {expected_ops['h']} H gate(s), found {ops.get('h', 0)}"

    assert "cx" in ops, "CNOT gate (cx) not found in the circuit."
    assert ops["cx"] == expected_ops["cx"], \
        f"Expected {expected_ops['cx']} CX gate(s), found {ops.get('cx', 0)}"

    assert "measure" in ops, "Measure operation not found in the circuit."
    assert ops["measure"] == expected_ops["measure"], \
        f"Expected {expected_ops['measure']} measure operations, found {ops.get('measure', 0)}"

    # Check total number of operations matches expected
    assert sum(ops.values()) == sum(expected_ops.values()), \
        f"Mismatch in total number of operations. Expected {sum(expected_ops.values())}, found {sum(ops.values())}"

def test_circuit_simulation_basic(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on the AerSimulator and performs basic checks on the results.
    Ensures simulation runs, returns counts, total counts match shots, and keys are valid bitstrings.
    """
    qc = quantum_circuit
    assert qc is not None, "QuantumCircuit fixture failed to load."
    logging.info(f"Running basic simulation test for circuit '{qc.name}' with {SHOTS} shots.")

    # Transpile for the simulator for potentially better performance/compatibility
    # Although AerSimulator often handles this well internally
    try:
        transpiled_qc = transpile(qc, simulator)
        logging.info("Circuit transpiled successfully for the simulator.")
    except Exception as e:
        logging.warning(f"Could not transpile circuit: {e}. Running original circuit.", exc_info=True)
        transpiled_qc = qc # Fallback to original circuit

    # Run the simulation
    job = simulator.run(transpiled_qc, shots=SHOTS)
    result = job.result()
    counts = result.get_counts(transpiled_qc) # Use the circuit object passed to run()

    logging.info(f"Simulation counts obtained: {counts}")

    # Assertions on the simulation results
    assert counts is not None, "Simulation did not return counts."
    assert isinstance(counts, dict), f"Counts should be a dictionary, but got {type(counts)}."
    assert sum(counts.values()) == SHOTS, \
        f"Total counts ({sum(counts.values())}) do not match the number of shots ({SHOTS})."

    # Check format of result keys (bitstrings)
    expected_bitstring_length = qc.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' is not a string."
        # Qiskit counts keys are typically reversed order compared to QASM declaration (c[1]c[0])
        # However, for Aer result.get_counts(), it often respects the order. Let's check length.
        assert len(bitstring) == expected_bitstring_length, \
            f"Bitstring '{bitstring}' has length {len(bitstring)}, expected {expected_bitstring_length} (matching classical bits)."
        assert all(c in '01' for c in bitstring), \
            f"Bitstring '{bitstring}' contains invalid characters (expected only '0' or '1')."

def test_bell_state_properties(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Tests the specific properties expected for a Bell state circuit (|00> + |11>)/sqrt(2).
    Checks that only the expected '00' and '11' states are measured (ideally) and
    that their probabilities are approximately equal (close to 0.5).
    """
    qc = quantum_circuit
    assert qc is not None, "QuantumCircuit fixture failed to load."
    logging.info(f"Testing Bell state properties for circuit '{qc.name}' with {SHOTS} shots.")

    # Transpile for the simulator
    try:
        transpiled_qc = transpile(qc, simulator)
        logging.info("Circuit transpiled successfully for Bell state test.")
    except Exception as e:
        logging.warning(f"Could not transpile circuit for Bell state test: {e}. Running original circuit.", exc_info=True)
        transpiled_qc = qc

    # Run the simulation
    job = simulator.run(transpiled_qc, shots=SHOTS)
    result = job.result()
    counts = result.get_counts(transpiled_qc)

    logging.info(f"Simulation counts for Bell state test: {counts}")
    assert counts is not None, "Simulation did not return counts for Bell state test."

    # Expected outcomes for the Bell state |Φ+> are '00' and '11'.
    # On an ideal simulator (like AerSimulator by default), other outcomes ('01', '10') should not appear.
    allowed_states = {'00', '11'}
    measured_states = set(counts.keys())

    logging.info(f"Measured states: {measured_states}")
    logging.info(f"Allowed states (ideal): {allowed_states}")

    # Check that only allowed states were measured
    unexpected_states = measured_states - allowed_states
    assert not unexpected_states, \
        f"Unexpected states measured: {unexpected_states}. Only {allowed_states} were expected for this Bell state circuit."

    # Check if both expected states are present (highly likely for sufficient shots)
    assert '00' in counts, f"Expected state '00' not found in counts. Counts: {counts}"
    assert '11' in counts, f"Expected state '11' not found in counts. Counts: {counts}"

    # Check if probabilities are roughly equal (allowing for statistical noise)
    # Define a tolerance for deviation from the ideal 0.5 probability
    prob_tolerance = 0.05 # Allow 5% deviation

    prob_00 = counts.get('00', 0) / SHOTS
    prob_11 = counts.get('11', 0) / SHOTS

    logging.info(f"Measured probabilities: P(00) = {prob_00:.4f}, P(11) = {prob_11:.4f}")

    assert abs(prob_00 - 0.5) < prob_tolerance, \
        f"Probability of '00' ({prob_00:.4f}) deviates significantly from the expected 0.5 (tolerance={prob_tolerance})."
    assert abs(prob_11 - 0.5) < prob_tolerance, \
        f"Probability of '11' ({prob_11:.4f}) deviates significantly from the expected 0.5 (tolerance={prob_tolerance})."

# --- Main execution check (optional) ---
if __name__ == "__main__":
    # This block is optional and useful for debugging the script directly.
    # However, the standard way to run these tests is using the pytest command.
    logging.info("This script contains Pytest tests. Run with 'pytest <script_name.py>'")
    # Example of how you might manually invoke pytest (requires pytest installed)
    # pytest.main(['-v', __file__])
