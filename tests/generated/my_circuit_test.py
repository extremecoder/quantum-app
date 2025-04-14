import pytest
import logging
import math
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- QASM Content ---
# Store the QASM content directly as a string, as the filename is not provided
# and this avoids potential path issues.
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
def simulator():
    """Provides a reusable AerSimulator instance."""
    logging.info("Setting up AerSimulator fixture.")
    return AerSimulator()

@pytest.fixture(scope="module")
def bell_circuit():
    """Loads the QuantumCircuit from the embedded QASM string."""
    try:
        logging.info("Loading QuantumCircuit from QASM string.")
        circuit = QuantumCircuit.from_qasm_str(QASM_CONTENT)
        logging.info(f"Circuit loaded successfully: {circuit.name}")
        return circuit
    except Exception as e:
        logging.error(f"Failed to load circuit from QASM string: {e}")
        pytest.fail(f"Failed to load circuit from QASM string: {e}")

# --- Test Functions ---

def test_circuit_structure(bell_circuit: QuantumCircuit):
    """
    Tests the basic structural properties of the loaded quantum circuit.
    """
    logging.info("Running test_circuit_structure...")
    assert bell_circuit is not None, "Circuit fixture should not be None"
    assert bell_circuit.num_qubits == 2, f"Expected 2 qubits, but got {bell_circuit.num_qubits}"
    logging.info(f"Verified number of qubits: {bell_circuit.num_qubits}")
    assert bell_circuit.num_clbits == 2, f"Expected 2 classical bits, but got {bell_circuit.num_clbits}"
    logging.info(f"Verified number of classical bits: {bell_circuit.num_clbits}")

    # Check for expected gate types
    op_names = {op.name for op, _, _ in bell_circuit.data}
    logging.info(f"Operations found in circuit: {op_names}")
    assert 'h' in op_names, "Circuit should contain an H gate"
    assert 'cx' in op_names, "Circuit should contain a CX gate"
    assert 'measure' in op_names, "Circuit should contain measure operations"
    logging.info("Verified presence of expected gate types (h, cx, measure).")

    # Check gate counts (optional, but good for specific circuits)
    assert bell_circuit.count_ops().get('h', 0) == 1, "Expected 1 H gate"
    assert bell_circuit.count_ops().get('cx', 0) == 1, "Expected 1 CX gate"
    assert bell_circuit.count_ops().get('measure', 0) == 2, "Expected 2 measure operations"
    logging.info("Verified counts of specific gates.")


def test_bell_state_simulation(bell_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the Bell state circuit on the simulator and checks the results.
    Expects roughly equal probabilities for |00> and |11> states.
    """
    logging.info("Running test_bell_state_simulation...")
    assert bell_circuit is not None, "Circuit fixture should not be None"
    assert simulator is not None, "Simulator fixture should not be None"

    shots = 4096  # Number of times to run the circuit
    logging.info(f"Simulating circuit '{bell_circuit.name}' for {shots} shots.")

    # Qiskit Aer recommends explicitly transpiling for the simulator
    # Although for basic gates it might not be strictly necessary
    try:
        transpiled_circuit = transpile(bell_circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=shots).result()
        counts = result.get_counts(transpiled_circuit)
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        pytest.fail(f"Simulation failed: {e}")

    logging.info(f"Simulation successful. Raw counts: {counts}")

    # 1. Check if counts were obtained
    assert counts is not None, "Simulation should produce counts."
    assert isinstance(counts, dict), "Counts should be a dictionary."

    # 2. Check total counts match shots
    total_counts = sum(counts.values())
    assert total_counts == shots, f"Total counts ({total_counts}) should equal shots ({shots})."
    logging.info(f"Verified total counts match shots ({total_counts}/{shots}).")

    # 3. Check format of result keys (bitstrings)
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' should be a string."
        assert len(bitstring) == bell_circuit.num_clbits, \
            f"Bitstring key '{bitstring}' length should match number of classical bits ({bell_circuit.num_clbits})."
        assert all(c in '01' for c in bitstring), f"Bitstring key '{bitstring}' should only contain '0' or '1'."
    logging.info(f"Verified format of result keys (expecting {bell_circuit.num_clbits}-bit strings).")

    # 4. Algorithm-Specific Checks (Bell State: |00> and |11>)
    expected_states = {'00', '11'}
    allowed_states = set(counts.keys())

    # Check that only the expected states are present (ideally)
    # Allow for small possibility of noise/errors if using noisy simulator, but AerSimulator is ideal
    assert allowed_states.issubset(expected_states), \
        f"Expected only states {expected_states}, but got {allowed_states}. Counts: {counts}"
    assert '00' in counts, "Expected state '00' to be present in results."
    assert '11' in counts, "Expected state '11' to be present in results."
    logging.info(f"Verified that observed states {allowed_states} are subset of expected {expected_states}.")

    # Check for roughly equal probabilities (allowing for statistical noise)
    # A simple check: each expected state should have at least some minimum fraction of shots
    # A more robust check might involve statistical tests (e.g., Chi-squared), but this is often sufficient.
    min_expected_fraction = 0.40 # Expecting ~0.5, allow significant deviation for smaller shot counts
    min_counts_per_state = int(shots * min_expected_fraction)

    count_00 = counts.get('00', 0)
    count_11 = counts.get('11', 0)

    logging.info(f"Counts for '00': {count_00} (~{count_00/shots*100:.2f}%)")
    logging.info(f"Counts for '11': {count_11} (~{count_11/shots*100:.2f}%)")

    assert count_00 > min_counts_per_state, \
        f"Count for '00' ({count_00}) is below threshold ({min_counts_per_state}) for {shots} shots."
    assert count_11 > min_counts_per_state, \
        f"Count for '11' ({count_11}) is below threshold ({min_counts_per_state}) for {shots} shots."
    logging.info(f"Verified that counts for '00' and '11' are roughly equal (above {min_expected_fraction*100}% threshold).")

# Example of how to run this test file:
# Save the code as e.g., `test_bell_circuit.py` in a `tests/` directory
# Make sure qiskit, qiskit-aer, and pytest are installed:
# `pip install qiskit qiskit-aer pytest`
# Run pytest from the terminal in the directory containing the `tests/` folder:
# `pytest`
