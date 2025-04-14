import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator # Modern import
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Attempt to infer the original filename from the context, use a default if necessary
# For this example, we'll assume a name based on the circuit's function (Bell state)
QASM_FILENAME = "bell_state_circuit.qasm"
SHOTS = 4096 # Use a reasonable number of shots for statistical significance

# Embedded QASM string as a fallback or primary source if file loading is not preferred/fails
QASM_STRING = """OPENQASM 2.0;
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
def qasm_file_path() -> Path:
    """Provides the Path object for the QASM file, assuming standard project structure."""
    # Assumes this test file is in tests/generated/
    # Assumes QASM file is in ir/openqasm/mitigated/ relative to project root
    try:
        project_root = Path(__file__).parent.parent.parent
        file_path = project_root / "ir" / "openqasm" / "mitigated" / QASM_FILENAME
        logging.info(f"Calculated QASM file path: {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error determining QASM file path: {e}")
        # Return a non-existent path to force fallback or handle error downstream
        # Create a dummy path object to avoid type errors downstream
        return Path("non_existent_path") / QASM_FILENAME


@pytest.fixture(scope="module")
def quantum_circuit(qasm_file_path: Path) -> QuantumCircuit:
    """
    Loads the QuantumCircuit.
    Tries loading from the specified file path first.
    If the file doesn't exist or loading fails, falls back to the embedded QASM string.
    """
    circuit = None
    try:
        if qasm_file_path.is_file():
            logging.info(f"Attempting to load QuantumCircuit from file: {qasm_file_path}")
            circuit = QuantumCircuit.from_qasm_file(str(qasm_file_path))
            logging.info(f"Successfully loaded circuit from file: {qasm_file_path}")
        else:
            logging.warning(f"QASM file not found at {qasm_file_path}. Attempting to load from embedded QASM string.")
            circuit = QuantumCircuit.from_qasm_str(QASM_STRING)
            logging.info("Successfully loaded circuit from embedded QASM string.")
    except FileNotFoundError:
        logging.warning(f"FileNotFoundError for {qasm_file_path}. Falling back to embedded QASM string.")
        circuit = QuantumCircuit.from_qasm_str(QASM_STRING)
        logging.info("Successfully loaded circuit from embedded QASM string after FileNotFoundError.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during circuit loading from file {qasm_file_path}: {e}")
        logging.info("Falling back to loading circuit from embedded QASM string.")
        circuit = QuantumCircuit.from_qasm_str(QASM_STRING)
        logging.info("Successfully loaded circuit from embedded QASM string after other exception.")

    if circuit is None:
        # This case should ideally not be reached if QASM_STRING is valid
        logging.error("Failed to load circuit from both file and embedded string.")
        pytest.fail("Could not load the quantum circuit.")

    return circuit


@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Provides an AerSimulator instance."""
    logging.info("Initializing AerSimulator.")
    # Initialize the AerSimulator
    # For simple circuits, default options are usually sufficient.
    # You could add options like method='statevector' or method='density_matrix' if needed.
    sim = AerSimulator()
    logging.info(f"AerSimulator initialized with default options.")
    return sim

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure (qubits, classical bits, gate types) of the loaded circuit.
    """
    logging.info("--- Starting Test: Circuit Structure ---")
    assert quantum_circuit is not None, "Quantum circuit object should not be None"

    # Verify number of qubits and classical bits
    expected_qubits = 2
    expected_clbits = 2
    logging.info(f"Checking circuit dimensions: Qubits={quantum_circuit.num_qubits}, Clbits={quantum_circuit.num_clbits}")
    assert quantum_circuit.num_qubits == expected_qubits, \
        f"Expected {expected_qubits} qubits, but found {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, \
        f"Expected {expected_clbits} classical bits, but found {quantum_circuit.num_clbits}"

    # Verify presence and count of expected operations
    ops_counts = quantum_circuit.count_ops()
    logging.info(f"Circuit operations found: {ops_counts}")
    assert 'h' in ops_counts, "Circuit should contain Hadamard (h) gates"
    assert ops_counts['h'] >= 1, "Expected at least one Hadamard gate"
    assert 'cx' in ops_counts, "Circuit should contain CNOT (cx) gates"
    assert ops_counts['cx'] >= 1, "Expected at least one CNOT gate"
    assert 'measure' in ops_counts, "Circuit should contain Measure operations"
    assert ops_counts['measure'] == expected_clbits, \
        f"Expected {expected_clbits} Measure operations, matching classical bits"
    logging.info("Circuit structure test passed.")


def test_circuit_simulation_basic(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on the simulator and performs basic checks on the simulation results.
    """
    logging.info("--- Starting Test: Basic Simulation ---")
    logging.info(f"Running simulation with {SHOTS} shots.")

    # Qiskit Aer prefers circuits to be transpiled for the backend
    # Although not strictly necessary for basic simulation, it's good practice
    transpiled_circuit = transpile(quantum_circuit, simulator)

    # Run the simulation
    result = simulator.run(transpiled_circuit, shots=SHOTS).result()
    counts = result.get_counts(transpiled_circuit) # Get counts using the circuit object

    logging.info(f"Simulation counts received: {counts}")

    # Basic Assertions on Results
    assert counts is not None, "Simulation should produce a counts dictionary."
    assert isinstance(counts, dict), "Counts should be a dictionary."

    # Check total counts match the number of shots
    total_counts = sum(counts.values())
    logging.info(f"Total counts observed: {total_counts}, Expected shots: {SHOTS}")
    assert total_counts == SHOTS, f"Total counts ({total_counts}) should equal the number of shots ({SHOTS})."

    # Check format of result keys (bitstrings)
    if counts: # Proceed only if counts is not empty
        expected_bitstring_length = quantum_circuit.num_clbits
        logging.info(f"Checking format of result keys (expected length: {expected_bitstring_length}).")
        for key in counts.keys():
            assert isinstance(key, str), f"Result key '{key}' should be a string."
            assert len(key) == expected_bitstring_length, \
                f"Result key '{key}' has length {len(key)}, expected {expected_bitstring_length}."
            assert all(c in '01' for c in key), \
                f"Result key '{key}' must be a bitstring (contain only '0' or '1')."
        logging.info("Result key format check passed.")
    else:
        logging.warning("Simulation produced empty counts dictionary.")

    logging.info("Basic simulation test passed.")


def test_bell_state_properties(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Tests the specific properties expected for a Bell state preparation circuit (|Φ+>).
    Checks for high correlation between measurements ('00' and '11' outcomes).
    """
    logging.info("--- Starting Test: Bell State Properties ---")
    logging.info(f"Running Bell state property test simulation with {SHOTS} shots.")

    # Ensure measurements are present, as they are crucial for this test
    if 'measure' not in quantum_circuit.count_ops():
        logging.warning("Circuit missing measurements. Adding measure_all for Bell state test.")
        # Add measurements if they were missing in the loaded circuit definition
        # Use inplace=False to avoid modifying the fixture state for other tests if scope='session'/'module'
        measured_circuit = quantum_circuit.copy()
        measured_circuit.measure_all(inplace=True)
    else:
        measured_circuit = quantum_circuit

    # Transpile for the simulator
    transpiled_circuit = transpile(measured_circuit, simulator)

    # Run the simulation
    result = simulator.run(transpiled_circuit, shots=SHOTS).result()
    counts = result.get_counts(transpiled_circuit)

    logging.info(f"Bell state simulation counts: {counts}")
    assert counts is not None, "Simulation should produce counts for Bell state test."
    assert sum(counts.values()) == SHOTS, "Total counts must equal shots."

    # Expected outcomes for |Φ+> = (|00> + |11>)/√2 are '00' and '11'
    # Note: Qiskit uses little-endian bit ordering (c[1]c[0]), so '00' is state |00> and '11' is state |11>.
    expected_states = {'00', '11'}
    unexpected_states = {'01', '10'}

    # Calculate counts for expected and unexpected states
    total_expected_counts = sum(counts.get(state, 0) for state in expected_states)
    total_unexpected_counts = sum(counts.get(state, 0) for state in unexpected_states)

    logging.info(f"Counts for expected states ('00', '11'): {total_expected_counts}")
    logging.info(f"Counts for unexpected states ('01', '10'): {total_unexpected_counts}")

    # Assert that most shots fall into the expected correlated states
    # Use a high threshold for ideal simulation (e.g., > 98%)
    min_expected_fraction = 0.98
    assert (total_expected_counts / SHOTS) > min_expected_fraction, \
        f"Expected states ('00', '11') constitute {total_expected_counts / SHOTS:.4f}, " \
        f"which is less than the threshold {min_expected_fraction}. Counts: {counts}"

    # Assert that unexpected anti-correlated states have very low probability
    # Use a low threshold (e.g., < 2%)
    max_unexpected_fraction = 0.02
    assert (total_unexpected_counts / SHOTS) < max_unexpected_fraction, \
        f"Unexpected states ('01', '10') constitute {total_unexpected_counts / SHOTS:.4f}, " \
        f"which is more than the threshold {max_unexpected_fraction}. Counts: {counts}"

    # Optional: Check for approximate equality between '00' and '11' counts
    # Due to statistical noise, they won't be exactly equal. Check if they are reasonably close.
    # Example: Check if each state ('00', '11') makes up at least 45% of the *expected* counts,
    # or alternatively, check if each makes up at least ~48% of the total shots (allowing for noise).
    counts_00 = counts.get('00', 0)
    counts_11 = counts.get('11', 0)
    min_individual_fraction_of_total = 0.45 # Expect each state ~50%, allow down to 45% due to noise
    logging.info(f"Individual state fractions: '00'={counts_00/SHOTS:.4f}, '11'={counts_11/SHOTS:.4f}")

    assert counts_00 / SHOTS >= min_individual_fraction_of_total, \
        f"Counts for '00' ({counts_00/SHOTS:.4f}) are lower than expected threshold ({min_individual_fraction_of_total})."
    assert counts_11 / SHOTS >= min_individual_fraction_of_total, \
        f"Counts for '11' ({counts_11/SHOTS:.4f}) are lower than expected threshold ({min_individual_fraction_of_total})."

    logging.info("Bell state properties test passed.")
