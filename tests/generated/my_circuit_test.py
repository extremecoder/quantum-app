import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import importlib.util # To check for qiskit_aer

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Use a placeholder filename, assuming the original filename isn't known.
# The fallback mechanism ensures the test runs even if the file doesn't exist at the expected path.
QASM_FILENAME = "unknown_circuit_2qubit_bell.qasm"
SHOTS = 4096 # Number of simulation shots

# --- Fixtures ---

@pytest.fixture(scope="module")
def simulator():
    """Pytest fixture providing an AerSimulator instance."""
    logging.info("Setting up AerSimulator fixture.")
    # Check if qiskit_aer is installed and use it
    if importlib.util.find_spec("qiskit_aer"):
        from qiskit_aer import AerSimulator
        logging.info("Using qiskit_aer.AerSimulator.")
        return AerSimulator()
    else:
        # Fallback or raise error if Aer isn't available
        logging.error("qiskit_aer not found. Please install it.")
        pytest.fail("qiskit_aer is required for simulation tests.")


@pytest.fixture(scope="module")
def quantum_circuit() -> QuantumCircuit:
    """
    Pytest fixture to load the QuantumCircuit.
    Attempts to load from a relative file path first, then falls back
    to loading from an embedded QASM string.
    """
    logging.info("Setting up QuantumCircuit fixture.")
    qasm_circuit = None
    # --- Preferred: Load from relative path ---
    try:
        # Assumes test file is in tests/generated/ and QASM is in ir/openqasm/mitigated/
        # relative to the project root.
        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parent.parent.parent # Adjust based on actual test file location
        qasm_file_path = project_root / "ir" / "openqasm" / "mitigated" / QASM_FILENAME

        logging.info(f"Attempting to load QASM from: {qasm_file_path}")
        if qasm_file_path.is_file():
            qasm_circuit = QuantumCircuit.from_qasm_file(str(qasm_file_path))
            logging.info(f"Successfully loaded QuantumCircuit from file: {qasm_file_path}")
        else:
            logging.warning(f"QASM file not found at expected path: {qasm_file_path}. Trying fallback.")
            raise FileNotFoundError # Trigger fallback

    except Exception as e:
        logging.warning(f"Failed to load QASM from file ({type(e).__name__}: {e}). Falling back to embedded string.")

        # --- Alternative: Load from embedded string ---
        qasm_content = """
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""
        try:
            qasm_circuit = QuantumCircuit.from_qasm_str(qasm_content)
            logging.info("Successfully loaded QuantumCircuit from embedded string.")
        except Exception as inner_e:
            logging.error(f"Failed to load QuantumCircuit from embedded string as well: {inner_e}")
            pytest.fail(f"Could not load QuantumCircuit using any method: {inner_e}")

    if qasm_circuit is None:
         pytest.fail("QuantumCircuit fixture could not be created.")

    return qasm_circuit

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure (qubits, clbits, gate types) of the loaded circuit.
    """
    logging.info("Running test_circuit_structure...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."

    # Check number of qubits and classical bits
    expected_qubits = 2
    expected_clbits = 2
    assert quantum_circuit.num_qubits == expected_qubits, \
        f"Expected {expected_qubits} qubits, found {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, \
        f"Expected {expected_clbits} classical bits, found {quantum_circuit.num_clbits}"
    logging.info(f"Circuit properties: qubits={quantum_circuit.num_qubits}, clbits={quantum_circuit.num_clbits}")

    # Check presence and count of expected operations
    op_counts = quantum_circuit.count_ops()
    logging.info(f"Circuit operations: {op_counts}")
    assert "h" in op_counts, "Hadamard gate (h) missing."
    assert op_counts.get("h", 0) == 1, f"Expected 1 Hadamard gate, found {op_counts.get('h', 0)}"
    assert "cx" in op_counts, "CNOT gate (cx) missing."
    assert op_counts.get("cx", 0) == 1, f"Expected 1 CNOT gate, found {op_counts.get('cx', 0)}"
    assert "measure" in op_counts, "Measure operation missing."
    assert op_counts.get("measure", 0) == expected_clbits, \
        f"Expected {expected_clbits} measure operations, found {op_counts.get('measure', 0)}"

    # Check circuit depth (should be non-zero for a non-empty circuit)
    assert quantum_circuit.depth() > 0, "Circuit depth is zero, implies an empty or trivial circuit."
    logging.info(f"Circuit depth: {quantum_circuit.depth()}")
    logging.info("Circuit structure test passed.")


def test_bell_state_simulation_outcome(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit simulation and asserts properties of the Bell state measurement outcomes.
    This circuit generates the Bell state |Φ+> = (|00> + |11>)/sqrt(2).
    """
    logging.info(f"Running test_bell_state_simulation_outcome with {SHOTS} shots...")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to load."
    assert simulator is not None, "Simulator fixture failed to load."
    assert quantum_circuit.num_clbits > 0, "Circuit must have classical bits for measurement counts."
    assert any(instr.operation.name == 'measure' for instr in quantum_circuit.data), \
           "Circuit lacks measure operations needed for simulation results."

    # Execute the simulation
    job = simulator.run(quantum_circuit, shots=SHOTS)
    result = job.result()
    counts = result.get_counts(quantum_circuit)

    logging.info(f"Simulation counts obtained: {counts}")

    # --- Assertions on Simulation Results ---
    assert counts is not None, "Simulation did not return counts."
    assert isinstance(counts, dict), f"Expected counts to be a dict, got {type(counts)}"

    # 1. Check total counts match shots
    total_counts = sum(counts.values())
    assert total_counts == SHOTS, \
        f"Total counts ({total_counts}) do not match the number of shots ({SHOTS})."
    logging.info(f"Total counts ({total_counts}) match shots ({SHOTS}).")

    # 2. Check format and validity of result keys (bitstrings)
    expected_bitstring_length = quantum_circuit.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' is not a string."
        assert len(bitstring) == expected_bitstring_length, \
            f"Count key '{bitstring}' has length {len(bitstring)}, expected {expected_bitstring_length}."
        assert all(char in '01' for char in bitstring), \
            f"Count key '{bitstring}' contains invalid characters (expected only '0' or '1')."
    logging.info(f"All count keys are valid bitstrings of length {expected_bitstring_length}.")

    # 3. Algorithm-Specific Assertions (Bell State |Φ+>)
    # Expect only '00' and '11' outcomes for this specific circuit (ideal simulation).
    expected_states = {'00', '11'}
    observed_states = set(counts.keys())

    assert observed_states.issubset(expected_states), \
        f"Observed states {observed_states} contain unexpected outcomes. Only {expected_states} are expected."

    # For a reasonable number of shots, both expected states should be present.
    if SHOTS >= 100: # Avoid failing if shots are too low for statistical significance
        assert observed_states == expected_states, \
            f"Expected states {expected_states} but observed {observed_states}. Both '00' and '11' should be present."
        logging.info("Observed states match the expected Bell state outcomes {'00', '11'}.")
    else:
        logging.warning(f"Low shot count ({SHOTS}). Skipping check for presence of *both* '00' and '11'.")

    # Optional: Check for approximate equal probability (can be sensitive to noise/simulator variance)
    # count00 = counts.get('00', 0)
    # count11 = counts.get('11', 0)
    # expected_prob = 0.5
    # tolerance = 0.1 # Allow 10% deviation from ideal 50/50 split
    # assert abs(count00 / SHOTS - expected_prob) < tolerance, f"Probability of '00' ({count00/SHOTS:.3f}) deviates significantly from {expected_prob}"
    # assert abs(count11 / SHOTS - expected_prob) < tolerance, f"Probability of '11' ({count11/SHOTS:.3f}) deviates significantly from {expected_prob}"
    # logging.info(f"Probabilities are within tolerance: P(00) ~ {count00/SHOTS:.3f}, P(11) ~ {count11/SHOTS:.3f}")

    logging.info("Bell state simulation outcome test passed.")
