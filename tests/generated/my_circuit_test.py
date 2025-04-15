import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator  # Modern import
from qiskit.result import Counts
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Test Configuration ---

# Using embedded QASM content as the primary strategy for robustness
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

# Simulation parameters
SHOTS = 4096
# Tolerance for checking probability distribution (e.g., 50% +/- 10%)
PROBABILITY_TOLERANCE = 0.10

# --- Fixtures ---

@pytest.fixture(scope="module")
def quantum_circuit() -> QuantumCircuit:
    """Fixture to load the QuantumCircuit from the embedded QASM string."""
    try:
        circuit = QuantumCircuit.from_qasm_str(QASM_CONTENT)
        logging.info("Successfully loaded QuantumCircuit from QASM string.")
        # Attempt to give the circuit a name for better logging/debugging
        circuit.name = "UnknownCircuit2Q_BellState"
        return circuit
    except Exception as e:
        logging.error(f"Failed to load QuantumCircuit from QASM string: {e}")
        pytest.fail(f"Failed to load QuantumCircuit: {e}")

@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Fixture to provide an AerSimulator instance."""
    logging.info("Initializing AerSimulator.")
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure of the loaded quantum circuit.
    Checks number of qubits, classical bits, and presence of key operations.
    """
    test_name = "test_circuit_structure"
    logging.info(f"--- Running {test_name} ---")
    assert quantum_circuit is not None, "Quantum circuit fixture failed to load."

    # Verify qubit and classical bit counts
    expected_qubits = 2
    expected_clbits = 2
    assert quantum_circuit.num_qubits == expected_qubits, \
        f"Expected {expected_qubits} qubits, found {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, \
        f"Expected {expected_clbits} classical bits, found {quantum_circuit.num_clbits}"
    logging.info(f"Circuit properties: num_qubits={quantum_circuit.num_qubits}, num_clbits={quantum_circuit.num_clbits}")

    # Verify presence of expected gate types
    op_counts = quantum_circuit.count_ops()
    expected_ops = {'h', 'cx', 'measure'}
    present_ops = set(op_counts.keys())
    assert expected_ops.issubset(present_ops), \
        f"Missing expected operations. Expected: {expected_ops}, Found: {present_ops}"
    logging.info(f"Operations found: {op_counts}")

    # Specific counts check (optional but useful here)
    assert op_counts.get('h', 0) == 1, f"Expected 1 H gate, found {op_counts.get('h', 0)}"
    assert op_counts.get('cx', 0) == 1, f"Expected 1 CX gate, found {op_counts.get('cx', 0)}"
    assert op_counts.get('measure', 0) == 2, f"Expected 2 Measure operations, found {op_counts.get('measure', 0)}"

    logging.info(f"--- {test_name} PASSED ---")


def test_circuit_simulation_basic(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on a simulator and performs basic checks on the results.
    Verifies counts object, total shots, and result key format.
    """
    test_name = "test_circuit_simulation_basic"
    logging.info(f"--- Running {test_name} ---")
    assert quantum_circuit is not None, "Quantum circuit fixture failed"
    assert simulator is not None, "Simulator fixture failed"

    # Ensure the circuit has classical bits for measurement results
    if quantum_circuit.num_clbits == 0:
         pytest.skip("Skipping simulation test: circuit has no classical bits defined.")

    try:
        logging.info(f"Running simulation with {SHOTS} shots...")
        job = simulator.run(quantum_circuit, shots=SHOTS)
        result = job.result()
        counts = result.get_counts(quantum_circuit)
        logging.info(f"Simulation finished. Counts obtained: {counts}")

        # Basic assertions on the counts object
        assert counts is not None, "Simulation should return a counts dictionary."
        assert isinstance(counts, dict), f"Counts should be a dictionary, but got {type(counts)}."
        assert len(counts) > 0, "Counts dictionary should not be empty after simulation."

        # Verify total counts match the number of shots
        total_counts = sum(counts.values())
        assert total_counts == SHOTS, \
            f"Total counts ({total_counts}) do not match the number of shots ({SHOTS})."
        logging.info(f"Verified total counts match shots ({SHOTS}).")

        # Check format of result keys (bitstrings)
        expected_bitstring_length = quantum_circuit.num_clbits
        for bitstring in counts.keys():
            assert isinstance(bitstring, str), f"Count key '{bitstring}' is not a string."
            # Qiskit counts keys are typically ordered c[n-1]...c[0]
            assert len(bitstring) == expected_bitstring_length, \
                f"Bitstring key '{bitstring}' has length {len(bitstring)}, expected {expected_bitstring_length} (number of classical bits)."
            assert all(c in '01' for c in bitstring), \
                f"Bitstring key '{bitstring}' contains invalid characters (should only be '0' or '1')."
        logging.info(f"Verified format of {len(counts)} result bitstring keys.")

        logging.info(f"--- {test_name} PASSED ---")

    except Exception as e:
        logging.error(f"Simulation or result validation failed in {test_name}: {e}", exc_info=True)
        pytest.fail(f"Simulation or result validation failed: {e}")


def test_bell_state_properties(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Tests the specific properties expected for the Bell state circuit (|00> + |11>)/sqrt(2).
    Checks for the presence and near-equal probability of '00' and '11' states.
    """
    test_name = "test_bell_state_properties"
    logging.info(f"--- Running {test_name} ---")
    assert quantum_circuit is not None, "Quantum circuit fixture failed"
    assert simulator is not None, "Simulator fixture failed"

    # This test is specific to the Bell state generated by H(q0); CX(q0, q1); Measure all
    if quantum_circuit.num_qubits != 2 or quantum_circuit.num_clbits != 2:
         pytest.skip("Skipping Bell state test: circuit does not have exactly 2 qubits and 2 classical bits.")

    try:
        logging.info(f"Running simulation for Bell state analysis with {SHOTS} shots...")
        job = simulator.run(quantum_circuit, shots=SHOTS)
        result = job.result()
        counts = result.get_counts(quantum_circuit)
        logging.info(f"Simulation for Bell state test finished. Counts: {counts}")

        assert sum(counts.values()) == SHOTS, "Total counts should equal shots."

        # Expected outcomes for |Φ+> = (|00> + |11>)/√2 are '00' and '11'
        # Qiskit's bit order is c[n-1]...c[0], so c[1]c[0] -> '00' and '11'
        expected_states = {'00', '11'}
        observed_states = set(counts.keys())

        # Check 1: The observed states should ideally only be the expected Bell states.
        # Allow for small noise by checking if unexpected states have very low probability.
        unexpected_counts = 0
        for state, count in counts.items():
            if state not in expected_states:
                logging.warning(f"Found unexpected state '{state}' with count {count}")
                unexpected_counts += count

        max_unexpected_prob = 0.02 # Allow up to 2% combined probability for noise states
        assert unexpected_counts / SHOTS <= max_unexpected_prob, \
            f"Probability of unexpected states ({unexpected_counts / SHOTS:.4f}) exceeds tolerance ({max_unexpected_prob})."

        # Check 2: Both expected states '00' and '11' must be present (unless shots are very low or noise is extreme)
        assert '00' in observed_states, "Expected state '00' not found in results."
        assert '11' in observed_states, "Expected state '11' not found in results."

        # Check 3: Probabilities of '00' and '11' should be roughly equal (around 0.5)
        prob_00 = counts.get('00', 0) / SHOTS
        prob_11 = counts.get('11', 0) / SHOTS
        logging.info(f"Observed probabilities: P(00)={prob_00:.4f}, P(11)={prob_11:.4f}")

        assert abs(prob_00 - 0.5) < PROBABILITY_TOLERANCE, \
            f"Probability of '00' ({prob_00:.4f}) is not within tolerance {PROBABILITY_TOLERANCE} of 0.5"
        assert abs(prob_11 - 0.5) < PROBABILITY_TOLERANCE, \
            f"Probability of '11' ({prob_11:.4f}) is not within tolerance {PROBABILITY_TOLERANCE} of 0.5"

        # Check 4: The sum of probabilities for the expected states should be close to 1.0
        # This is implicitly covered by Check 1 and Check 3, but can be asserted explicitly
        total_expected_prob = prob_00 + prob_11
        assert abs(total_expected_prob - 1.0) < max_unexpected_prob + 0.01, \
             f"Sum of probabilities for '00' and '11' ({total_expected_prob:.4f}) deviates significantly from 1.0"

        logging.info("Bell state properties (correlation '00'/'11' and ~equal probabilities) verified.")
        logging.info(f"--- {test_name} PASSED ---")

    except Exception as e:
        logging.error(f"Bell state property test failed in {test_name}: {e}", exc_info=True)
        pytest.fail(f"Bell state property test failed: {e}")
