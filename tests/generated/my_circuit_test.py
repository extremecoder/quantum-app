import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit
# Use modern qiskit_aer import
from qiskit_aer import AerSimulator

# Configure basic logging for the test module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- QASM Content ---
# Using embedded string as the alternative strategy for robustness,
# as the exact relative path setup can be environment-dependent.
# Assumed original filename: bell_state_measure.qasm (for context only)
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

# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def qasm_content() -> str:
    """Provides the QASM circuit content as a string."""
    logging.info("Providing QASM content fixture.")
    return QASM_CONTENT

@pytest.fixture(scope="module")
def quantum_circuit(qasm_content: str) -> QuantumCircuit:
    """
    Loads the QuantumCircuit from the QASM string content.
    Uses module scope for efficiency, loading the circuit only once per module.
    """
    logging.info("Loading QuantumCircuit from QASM string fixture.")
    try:
        circuit = QuantumCircuit.from_qasm_str(qasm_content)
        logging.info("QuantumCircuit loaded successfully.")
        # Log basic circuit properties upon loading
        logging.info(f"Circuit - Qubits: {circuit.num_qubits}, Clbits: {circuit.num_clbits}, Depth: {circuit.depth()}, Ops: {circuit.count_ops()}")
        return circuit
    except Exception as e:
        logging.error(f"Failed to load QuantumCircuit from QASM string: {e}", exc_info=True)
        pytest.fail(f"Failed to load QuantumCircuit from QASM string: {e}")

@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Provides an AerSimulator instance for running simulations."""
    logging.info("Creating AerSimulator fixture.")
    # Initialize the AerSimulator
    # You could add options here, e.g., method='statevector' or noise_model
    sim = AerSimulator()
    logging.info(f"AerSimulator backend: {sim.configuration().backend_name}")
    return sim

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure of the loaded quantum circuit.
    Checks qubit count, classical bit count, and presence/count of key operations.
    """
    logging.info("--- Starting test_circuit_structure ---")
    assert quantum_circuit is not None, "QuantumCircuit object should be loaded."

    # Verify qubit and classical bit counts
    expected_qubits = 2
    expected_clbits = 2
    assert quantum_circuit.num_qubits == expected_qubits, \
        f"Expected {expected_qubits} qubits, but found {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, \
        f"Expected {expected_clbits} classical bits, but found {quantum_circuit.num_clbits}"
    logging.info(f"Verified: {quantum_circuit.num_qubits} qubits, {quantum_circuit.num_clbits} classical bits.")

    # Verify operation counts
    op_counts = quantum_circuit.count_ops()
    logging.info(f"Operations found: {op_counts}")
    expected_ops = {'h': 1, 'cx': 1, 'measure': 2}
    total_expected_ops = sum(expected_ops.values())

    assert 'h' in op_counts, "Hadamard (h) gate missing."
    assert op_counts['h'] == expected_ops['h'], f"Expected {expected_ops['h']} 'h' gate(s), found {op_counts['h']}"

    assert 'cx' in op_counts, "CNOT (cx) gate missing."
    assert op_counts['cx'] == expected_ops['cx'], f"Expected {expected_ops['cx']} 'cx' gate(s), found {op_counts['cx']}"

    assert 'measure' in op_counts, "Measure operation missing."
    assert op_counts['measure'] == expected_ops['measure'], f"Expected {expected_ops['measure']} 'measure' ops, found {op_counts['measure']}"

    # Verify total number of operations
    assert sum(op_counts.values()) == total_expected_ops, \
        f"Expected total {total_expected_ops} operations, found {sum(op_counts.values())}"

    logging.info("Circuit structure verification passed.")
    logging.info("--- Finished test_circuit_structure ---")


def test_circuit_simulation_bell_state(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on an ideal simulator and verifies the expected Bell state outcomes.
    The circuit creates (|00> + |11>)/sqrt(2), so measurements should yield '00' or '11'.
    """
    logging.info("--- Starting test_circuit_simulation_bell_state ---")
    shots = 4096  # A reasonable number of shots for statistical significance

    # Run the simulation
    logging.info(f"Running simulation with {shots} shots...")
    # No explicit transpilation needed for AerSimulator with basic gates usually
    job = simulator.run(quantum_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(quantum_circuit)

    logging.info(f"Simulation completed. Result counts: {counts}")

    # --- Assertions on Simulation Results ---

    # 1. Basic checks
    assert counts is not None, "Simulation result should include counts."
    assert isinstance(counts, dict), "Counts should be a dictionary."
    assert sum(counts.values()) == shots, \
        f"Total counts ({sum(counts.values())}) do not match the number of shots ({shots})."

    # 2. Check format of result keys (bitstrings)
    expected_bitstring_length = quantum_circuit.num_clbits
    assert expected_bitstring_length == 2, "This test assumes 2 classical bits for result keys."
    if not counts:
        logging.warning("Counts dictionary is empty. Cannot perform further checks.")
        # Fail if empty, as we expect results
        pytest.fail("Simulation produced no measurement counts.")

    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' is not a string."
        assert len(bitstring) == expected_bitstring_length, \
            f"Bitstring key '{bitstring}' has length {len(bitstring)}, expected {expected_bitstring_length}."
        assert all(c in '01' for c in bitstring), \
            f"Bitstring key '{bitstring}' contains invalid characters (should only be '0' or '1')."

    # 3. Algorithm-Specific Assertions (Bell State Verification)
    # For the state (|00> + |11>)/sqrt(2), we expect only '00' and '11' outcomes in an ideal simulation.
    allowed_outcomes = {'00', '11'}
    observed_outcomes = set(counts.keys())

    logging.info(f"Observed outcomes: {observed_outcomes}. Expected subset of: {allowed_outcomes}")

    # Check that only allowed outcomes were observed
    assert observed_outcomes.issubset(allowed_outcomes), \
        f"Observed unexpected outcomes: {observed_outcomes - allowed_outcomes}. Only {allowed_outcomes} are expected."

    # Check that *both* expected outcomes are present (highly likely with 4096 shots)
    # This confirms the superposition and entanglement led to both possibilities.
    assert '00' in counts, "Expected outcome '00' was not observed."
    assert '11' in counts, "Expected outcome '11' was not observed."

    # Optional: Check if probabilities are roughly equal (can be flaky, use approx)
    # prob_00 = counts.get('00', 0) / shots
    # prob_11 = counts.get('11', 0) / shots
    # logging.info(f"Observed probabilities: P(00)={prob_00:.4f}, P(11)={prob_11:.4f}")
    # assert prob_00 == pytest.approx(0.5, abs=0.05) # Check if P(00) is close to 0.5
    # assert prob_11 == pytest.approx(0.5, abs=0.05) # Check if P(11) is close to 0.5

    logging.info("Simulation results are consistent with the expected Bell state (|00> + |11>)/sqrt(2).")
    logging.info("--- Finished test_circuit_simulation_bell_state ---")

# Example of how to run this test:
# 1. Save this code as a Python file (e.g., test_bell_circuit.py) in your tests directory.
# 2. Make sure you have pytest and qiskit-aer installed (`pip install pytest qiskit qiskit-aer`).
# 3. Run pytest from your terminal in the project's root directory: `pytest`
