import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.result import Result
from qiskit_aer import AerSimulator  # Modern import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Fixtures ---

@pytest.fixture(scope="module")
def qasm_content() -> str:
    """Provides the QASM content as a string (alternative loading strategy)."""
    # This approach embeds the QASM directly, avoiding potential path issues.
    return """OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""

@pytest.fixture(scope="module")
def quantum_circuit(qasm_content: str) -> QuantumCircuit:
    """
    Loads the QuantumCircuit from the QASM string provided by the qasm_content fixture.
    """
    logging.info("Loading QuantumCircuit from QASM string.")
    try:
        circuit = QuantumCircuit.from_qasm_str(qasm_content)
        assert circuit is not None
        logging.info("QuantumCircuit loaded successfully from string.")
        # Log basic properties upon loading
        logging.info(f"Circuit properties: Name='{circuit.name}', Qubits={circuit.num_qubits}, Clbits={circuit.num_clbits}")
        # logging.info(f"Circuit instructions:\n{circuit.draw(output='text')}") # Optional: log circuit diagram
        return circuit
    except Exception as e:
        logging.error(f"Failed to load QuantumCircuit from string: {e}")
        pytest.fail(f"Failed to load QuantumCircuit from string: {e}", pytrace=False)

@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Provides an AerSimulator instance."""
    logging.info("Initializing AerSimulator.")
    # You could configure simulator options here if needed (e.g., method='statevector')
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure of the loaded quantum circuit.
    Checks number of qubits, classical bits, and presence of key operations.
    """
    logging.info("Running test_circuit_structure...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."

    # Check qubit and classical bit counts
    expected_qubits = 2
    expected_clbits = 2
    assert quantum_circuit.num_qubits == expected_qubits, \
        f"Expected {expected_qubits} qubits, but got {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, \
        f"Expected {expected_clbits} classical bits, but got {quantum_circuit.num_clbits}"
    logging.info(f"Verified: Qubits={quantum_circuit.num_qubits}, Clbits={quantum_circuit.num_clbits}")

    # Check for presence of expected operations by name
    op_names = {instr.operation.name for instr, _, _ in quantum_circuit.data}
    logging.info(f"Operations found in circuit: {op_names}")
    assert "h" in op_names, "Hadamard (h) gate not found in the circuit."
    assert "cx" in op_names, "Controlled-X (cx) gate not found in the circuit."
    assert "measure" in op_names, "Measure operation not found in the circuit."
    logging.info("Presence of key operations (h, cx, measure) verified.")

    logging.info("Circuit structure test passed.")


def test_circuit_simulation(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on a simulator and checks basic result properties like counts format.
    """
    logging.info("Running test_circuit_simulation...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."
    assert simulator is not None, "Simulator fixture failed to load."

    shots = 1024 # A standard number of shots for basic simulation testing
    logging.info(f"Running simulation with {shots} shots.")

    # Ensure the circuit has measurements if we expect counts
    if quantum_circuit.num_clbits == 0:
         pytest.skip("Skipping simulation test as the circuit has no classical bits for measurement results.")

    try:
        # Qiskit Aer's run method expects the circuit directly
        job = simulator.run(quantum_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(quantum_circuit)

        logging.info(f"Simulation counts obtained: {counts}")

        assert counts is not None, "Simulation did not return counts."
        assert isinstance(counts, dict), f"Counts should be a dictionary, but got {type(counts)}."
        assert sum(counts.values()) == shots, \
            f"Total counts ({sum(counts.values())}) do not match the number of shots ({shots})."

        # Check format of result keys (bitstrings)
        if counts: # Proceed only if there are counts
             first_key = next(iter(counts))
             expected_bitstring_length = quantum_circuit.num_clbits
             assert len(first_key) == expected_bitstring_length, \
                 f"Result keys (bitstrings) have incorrect length. Expected {expected_bitstring_length}, got {len(first_key)} for key '{first_key}'."
             for key in counts.keys():
                 assert all(c in '01' for c in key), f"Result key '{key}' contains invalid characters (should be 0 or 1)."
                 assert len(key) == expected_bitstring_length, \
                     f"Result key '{key}' has inconsistent length. Expected {expected_bitstring_length}."
             logging.info(f"Verified format of {len(counts)} result keys (length={expected_bitstring_length}, chars='0'/'1').")
        else:
             logging.warning("Simulation resulted in empty counts dictionary.") # Should not happen for this circuit with shots > 0

        logging.info("Basic simulation results structure test passed.")

    except Exception as e:
        logging.error(f"Simulation execution failed: {e}")
        pytest.fail(f"Simulation failed with exception: {e}", pytrace=False)


def test_bell_state_outcome(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Tests the specific outcome distribution expected from a Bell state preparation circuit.
    Specifically, expects only '00' and '11' outcomes in an ideal simulation.
    """
    logging.info("Running test_bell_state_outcome...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."
    assert simulator is not None, "Simulator fixture failed to load."

    # This circuit prepares the Bell state (|00> + |11>)/sqrt(2)
    # Ideal measurement should yield only '00' and '11' states.
    expected_states = {'00', '11'}

    shots = 4096 # Use enough shots for statistical relevance
    logging.info(f"Running Bell state simulation with {shots} shots, expecting outcomes {expected_states}.")

    if quantum_circuit.num_clbits != 2:
         pytest.skip(f"Skipping Bell state test as the circuit does not have 2 classical bits (found {quantum_circuit.num_clbits}).")

    try:
        job = simulator.run(quantum_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(quantum_circuit)
        logging.info(f"Bell state simulation counts: {counts}")

        assert counts is not None, "Simulation did not return counts for Bell state test."
        assert sum(counts.values()) == shots, f"Total counts ({sum(counts.values())}) do not match the number of shots ({shots})."

        measured_states = set(counts.keys())
        logging.info(f"Measured states: {measured_states}")

        # Primary Assertion: Only expected states should be present in the results.
        assert measured_states.issubset(expected_states), \
            f"Simulation produced unexpected states: {measured_states - expected_states}. Only {expected_states} were expected."

        # Secondary Assertion: Ensure the expected states are actually observed (with sufficient shots).
        # This checks that the simulation didn't just return an empty dictionary or only one state.
        if shots > 10: # Avoid failing if shots are pathologically low
            assert '00' in measured_states, f"Expected state '00' not found in simulation results (counts: {counts})."
            assert '11' in measured_states, f"Expected state '11' not found in simulation results (counts: {counts})."
            logging.info("Verified that both '00' and '11' states were observed.")
        else:
            logging.warning(f"Skipping check for presence of both '00' and '11' due to low shot count ({shots}).")

        logging.info("Bell state outcome test passed: Only expected states '00' and '11' were observed.")

    except Exception as e:
        logging.error(f"Bell state simulation failed: {e}")
        pytest.fail(f"Bell state simulation failed with exception: {e}", pytrace=False)

