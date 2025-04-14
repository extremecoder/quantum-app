import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.result import Counts
from qiskit.qasm2 import QASM2ImportError  # Specific import for QASM load errors

# Configure logging
# Ensure logs are visible when running pytest (e.g., pytest -s -v)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Assumed filename for the input QASM based on its content (Bell state)
# This filename will be used to construct the relative path.
QASM_FILENAME = "bell_state_circuit_2q.qasm"
# Number of shots for simulation
SHOTS = 4096
# Embedded QASM content as a fallback
FALLBACK_QASM_STRING = """OPENQASM 2.0;
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
def qasm_file_path() -> Path | None:
    """
    Provides the Path object for the QASM file.
    Tries to locate the file relative to the test file's location.
    Returns None if the file is not found.
    """
    # Construct the path relative to the test file location
    # Assumes test file is in tests/generated/ and QASM is in ir/openqasm/mitigated/
    try:
        test_file_dir = Path(__file__).parent
        project_root = test_file_dir.parent.parent # Go up two levels from tests/generated/
        file_path = project_root / "ir" / "openqasm" / "mitigated" / QASM_FILENAME
        logging.info(f"Calculated QASM file path: {file_path}")

        if file_path.is_file():
            logging.info(f"QASM file found at: {file_path}")
            return file_path
        else:
            logging.warning(f"QASM file not found at the expected path: {file_path}. Will attempt fallback.")
            return None
    except Exception as e:
        logging.error(f"Error constructing QASM file path: {e}")
        return None

@pytest.fixture(scope="module")
def quantum_circuit(qasm_file_path: Path | None) -> QuantumCircuit:
    """
    Loads the QuantumCircuit. Prefers loading from the file path,
    falls back to the embedded QASM string if the file is not found.
    Fails the test setup if loading is unsuccessful.
    """
    circuit = None
    if qasm_file_path:
        try:
            circuit = QuantumCircuit.from_qasm_file(str(qasm_file_path))
            logging.info(f"Successfully loaded QuantumCircuit from file: {qasm_file_path}")
            return circuit
        except QASM2ImportError as e:
            logging.error(f"Error loading QASM file {qasm_file_path}: {e}. Attempting fallback.")
        except FileNotFoundError:
            logging.error(f"FileNotFoundError for {qasm_file_path}, though it was checked. Attempting fallback.")
        except Exception as e:
            logging.error(f"An unexpected error occurred loading QASM file {qasm_file_path}: {e}. Attempting fallback.")

    # Fallback to embedded string if file loading failed or path was None
    if circuit is None:
        logging.warning(f"Attempting to load QuantumCircuit from embedded QASM string for {QASM_FILENAME}.")
        try:
            circuit = QuantumCircuit.from_qasm_str(FALLBACK_QASM_STRING)
            logging.info("Successfully loaded QuantumCircuit from embedded QASM string.")
            return circuit
        except QASM2ImportError as e:
            logging.error(f"Failed to load QuantumCircuit from embedded QASM string: {e}")
            pytest.fail(f"Failed to load circuit from both file and fallback string: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred loading from embedded QASM string: {e}")
            pytest.fail(f"Failed to load circuit from both file and fallback string: {e}")

    # This point should ideally not be reached if pytest.fail works, but added for safety
    if circuit is None:
        pytest.fail("Quantum circuit could not be loaded.")

    # Ensure return type consistency, though fail should prevent reaching here if None
    return circuit # type: ignore


@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Provides an AerSimulator instance."""
    logging.info("Initializing AerSimulator")
    # Using the default ideal simulator
    return AerSimulator()

# --- Test Functions ---

def test_circuit_loaded_successfully(quantum_circuit: QuantumCircuit):
    """Tests if the quantum circuit object was created."""
    logging.info("Running test: test_circuit_loaded_successfully")
    assert quantum_circuit is not None, "QuantumCircuit fixture returned None"
    assert isinstance(quantum_circuit, QuantumCircuit), "Loaded object is not a QuantumCircuit"
    logging.info(f"Circuit successfully loaded (Name: {quantum_circuit.name})")

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """Tests the basic structure (qubits, clbits, gate types) of the loaded circuit."""
    logging.info("Running test: test_circuit_structure")
    assert quantum_circuit is not None

    # Check number of qubits and classical bits
    expected_qubits = 2
    expected_clbits = 2
    logging.info(f"Checking circuit structure: Qubits={quantum_circuit.num_qubits}, Clbits={quantum_circuit.num_clbits}")
    assert quantum_circuit.num_qubits == expected_qubits, \
        f"Expected {expected_qubits} qubits, but found {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, \
        f"Expected {expected_clbits} classical bits, but found {quantum_circuit.num_clbits}"

    # Check presence and count of specific gates/operations
    op_counts = quantum_circuit.count_ops()
    logging.info(f"Circuit operations breakdown: {op_counts}")
    assert 'h' in op_counts, "Circuit should contain Hadamard (h) gates"
    assert op_counts['h'] == 1, f"Expected 1 Hadamard gate, found {op_counts.get('h', 0)}"
    assert 'cx' in op_counts, "Circuit should contain CNOT (cx) gates"
    assert op_counts['cx'] == 1, f"Expected 1 CNOT gate, found {op_counts.get('cx', 0)}"
    assert 'measure' in op_counts, "Circuit should contain Measure operations"
    assert op_counts['measure'] == 2, f"Expected 2 Measure operations, found {op_counts.get('measure', 0)}"

    # Check total number of instructions in the circuit data
    # QASM: h, cx, measure, measure = 4 instructions expected
    expected_instructions = 4
    assert len(quantum_circuit.data) == expected_instructions, \
        f"Expected {expected_instructions} instructions, but found {len(quantum_circuit.data)}"
    logging.info("Circuit structure test passed.")

def test_simulation_basic_run(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """Runs the circuit on the simulator and checks basic result validity."""
    logging.info("Running test: test_simulation_basic_run")
    assert quantum_circuit is not None
    assert simulator is not None

    logging.info(f"Executing circuit on {simulator.name()} with {SHOTS} shots.")
    try:
        # Execute the circuit
        job = simulator.run(quantum_circuit, shots=SHOTS)
        result = job.result()
        counts = result.get_counts(quantum_circuit)
    except Exception as e:
        logging.error(f"Simulation execution failed: {e}")
        pytest.fail(f"Simulation failed with exception: {e}")

    logging.info(f"Simulation finished. Raw counts: {counts}")

    # Basic assertions on the result counts
    assert counts is not None, "Simulation result's get_counts() returned None."
    assert isinstance(counts, Counts), f"Expected results type Counts, but got {type(counts)}"
    assert len(counts) > 0, "Simulation produced zero counts (no results)."

    # Check total number of shots matches the sum of counts
    total_counts = sum(counts.values())
    logging.info(f"Total counts obtained: {total_counts}, Expected shots: {SHOTS}")
    assert total_counts == SHOTS, f"Total counts ({total_counts}) does not match the number of shots ({SHOTS})"

    # Check format of result keys (bitstrings)
    expected_bitstring_length = quantum_circuit.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' is not a string."
        # Qiskit bitstring order is typically c[N-1]...c[1]c[0].
        # For `creg c[2]`, measured q[0]->c[0], q[1]->c[1], keys should be 'c1c0'.
        assert len(bitstring) == expected_bitstring_length, \
            f"Bitstring key '{bitstring}' has length {len(bitstring)}, expected {expected_bitstring_length}"
        assert all(c in '01' for c in bitstring), \
            f"Bitstring key '{bitstring}' contains invalid characters (expected only '0' or '1')."
    logging.info("Basic simulation run checks passed.")


def test_bell_state_entanglement_outcomes(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Checks if simulation outcomes reflect the expected Bell state entanglement.
    For the state (|00> + |11>)/sqrt(2), only '00' and '11' outcomes should appear
    in an ideal simulation.
    """
    logging.info("Running test: test_bell_state_entanglement_outcomes")
    assert quantum_circuit is not None
    assert simulator is not None

    logging.info(f"Executing Bell state circuit on {simulator.name()} with {SHOTS} shots.")
    try:
        job = simulator.run(quantum_circuit, shots=SHOTS)
        result = job.result()
        counts = result.get_counts(quantum_circuit)
    except Exception as e:
        logging.error(f"Simulation execution failed during Bell state test: {e}")
        pytest.fail(f"Simulation failed with exception: {e}")

    logging.info(f"Bell state simulation counts: {counts}")

    # Assertions for the Bell State (|00> + |11>)/sqrt(2)
    # Qiskit measures q[0]->c[0], q[1]->c[1]. Result keys are 'c1c0'.
    # Therefore, the expected outcomes are '00' and '11'.
    allowed_states = {'00', '11'}
    measured_states = set(counts.keys())

    logging.info(f"Expected states: {allowed_states}, Measured states: {measured_states}")

    # Check that only the allowed states were measured
    unexpected_states = measured_states - allowed_states
    assert not unexpected_states, \
        f"Simulation produced unexpected states: {unexpected_states}. Only {allowed_states} are expected for this Bell state circuit on an ideal simulator."

    # Check that both expected states are present (highly likely with sufficient shots)
    assert '00' in counts, f"Expected state '00' not found in simulation results (Counts: {counts})"
    assert '11' in counts, f"Expected state '11' not found in simulation results (Counts: {counts})"

    # Optional: Check for approximate equiprobability (within statistical noise)
    count_00 = counts.get('00', 0)
    count_11 = counts.get('11', 0)
    prob_00 = count_00 / SHOTS
    prob_11 = count_11 / SHOTS
    logging.info(f"Probabilities: P(00) = {prob_00:.4f}, P(11) = {prob_11:.4f}")
    # Allow for some deviation from 0.5 due to finite shots
    tolerance = 3 * (0.5 * 0.5 / SHOTS)**0.5 # 3-sigma deviation for binomial distribution
    assert abs(prob_00 - 0.5) < tolerance, \
        f"Probability of '00' ({prob_00:.4f}) deviates significantly from 0.5 (Tolerance: {tolerance:.4f})"
    assert abs(prob_11 - 0.5) < tolerance, \
        f"Probability of '11' ({prob_11:.4f}) deviates significantly from 0.5 (Tolerance: {tolerance:.4f})"

    logging.info("Bell state entanglement outcome test passed.")

