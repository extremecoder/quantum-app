import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.qasm2 import QASM2ImportError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the expected QASM filename based on the input circuit structure/purpose
# Since it's a simple 2-qubit entanglement circuit, let's name it appropriately.
QASM_FILENAME = "unknown_2q_entanglement_circuit.qasm"

# Define the QASM content provided in the prompt
# This will be used to create the file if it doesn't exist, or as a fallback.
QASM_CONTENT = """OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""

# Helper function to ensure QASM file exists for testing
def ensure_qasm_file_exists(qasm_path: Path, qasm_content: str):
    """Creates the QASM file if it doesn't exist."""
    if not qasm_path.exists():
        logging.warning(f"QASM file not found at {qasm_path}. Creating a dummy file for testing.")
        try:
            qasm_path.parent.mkdir(parents=True, exist_ok=True)
            with open(qasm_path, "w") as f:
                f.write(qasm_content)
            logging.info(f"Created dummy QASM file at {qasm_path}")
        except OSError as e:
            logging.error(f"Failed to create directory or file at {qasm_path}: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while creating {qasm_path}: {e}")
            raise # Reraise after logging

# Fixture for the QASM file path
@pytest.fixture(scope="module")
def qasm_file_path() -> Path:
    """Provides the path to the QASM file, ensuring it exists."""
    # Calculate the path relative to the test file location
    # Assumes test file is in tests/generated/test_*.py
    # Assumes QASM file is in ir/openqasm/mitigated/
    try:
        test_file_path = Path(__file__).resolve()
        project_root = test_file_path.parent.parent.parent
        qasm_path = project_root / "ir" / "openqasm" / "mitigated" / QASM_FILENAME
    except Exception as e:
        logging.error(f"Error constructing QASM file path: {e}")
        pytest.fail(f"Could not determine QASM file path relative to {__file__}")
        return None # Should not be reached due to pytest.fail

    # Ensure the file exists (create if necessary for the test run)
    ensure_qasm_file_exists(qasm_path, QASM_CONTENT)

    return qasm_path

# Fixture for the QuantumCircuit object loaded from file
@pytest.fixture(scope="module")
def quantum_circuit(qasm_file_path: Path) -> QuantumCircuit:
    """Loads the QuantumCircuit from the QASM file."""
    if not qasm_file_path or not qasm_file_path.is_file():
         pytest.fail(f"Invalid QASM file path provided or file does not exist: {qasm_file_path}")

    try:
        logging.info(f"Loading QuantumCircuit from: {qasm_file_path}")
        circuit = QuantumCircuit.from_qasm_file(str(qasm_file_path))
        logging.info(f"QuantumCircuit loaded successfully from {qasm_file_path.name}.")
        return circuit
    except FileNotFoundError:
        pytest.fail(f"QASM file not found at {qasm_file_path}. Ensure the file exists or the path is correct.")
    except QASM2ImportError as e:
        pytest.fail(f"Failed to parse QASM file {qasm_file_path}: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred while loading QASM file {qasm_file_path}: {e}")

# Fixture for the AerSimulator
@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Provides an instance of the AerSimulator."""
    logging.info("Initializing AerSimulator.")
    # Basic ideal simulator
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure (qubits, clbits, gate types) of the loaded quantum circuit.
    """
    logging.info(f"Running test_circuit_structure for {QASM_FILENAME}...")
    assert quantum_circuit is not None, "QuantumCircuit object should be loaded."

    # Verify qubit and classical bit counts
    expected_qubits = 2
    expected_clbits = 2
    assert quantum_circuit.num_qubits == expected_qubits, \
        f"Expected {expected_qubits} qubits, but found {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, \
        f"Expected {expected_clbits} classical bits, but found {quantum_circuit.num_clbits}"

    logging.info(f"Circuit properties: num_qubits={quantum_circuit.num_qubits}, num_clbits={quantum_circuit.num_clbits}, depth={quantum_circuit.depth()}")

    # Check for presence of expected gate types based on the QASM content
    gate_types = {instruction.operation.name for instruction in quantum_circuit.data}
    logging.info(f"Gate types found in circuit: {gate_types}")
    assert "h" in gate_types, "Hadamard (h) gate should be present."
    assert "cx" in gate_types, "CNOT (cx) gate should be present."
    assert "measure" in gate_types, "Measure operation should be present."

    # Check circuit is not empty
    assert quantum_circuit.depth() > 0, "Circuit depth should be greater than 0."
    assert len(quantum_circuit.data) > 0, "Circuit should contain operations."

    logging.info("test_circuit_structure passed.")

def test_simulation_results(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on a simulator and checks basic properties of the results.
    Specifically tests for Bell state (|Φ⁺>) characteristics.
    """
    logging.info(f"Running test_simulation_results for {QASM_FILENAME}...")
    shots = 4096 # Use a reasonable number of shots for statistical significance

    # Ensure measurements are present before running simulation
    assert quantum_circuit.num_clbits > 0, "Circuit must have classical bits for measurement results."
    has_measure = any(instr.operation.name == 'measure' for instr in quantum_circuit.data)
    assert has_measure, "Circuit must contain measure operations to produce counts."

    logging.info(f"Running simulation with {shots} shots using {simulator.configuration().backend_name}...")
    # Qiskit Aer's run method returns a job object, then get result() and get_counts()
    job = simulator.run(quantum_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(quantum_circuit)

    logging.info(f"Simulation counts obtained: {counts}")

    assert counts is not None, "Simulation should produce counts."
    assert isinstance(counts, dict), "Counts should be a dictionary."

    # 1. Check total counts
    total_counts = sum(counts.values())
    assert total_counts == shots, f"Total counts ({total_counts}) should equal the number of shots ({shots})."

    # 2. Check format of result keys (bitstrings)
    expected_bitstring_length = quantum_circuit.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' should be a string."
        assert len(bitstring) == expected_bitstring_length, \
            f"Bitstring '{bitstring}' length ({len(bitstring)}) should match number of classical bits ({expected_bitstring_length})."
        assert all(c in '01' for c in bitstring), \
            f"Bitstring '{bitstring}' should contain only '0' or '1'."

    # 3. Algorithm-Specific Checks: Bell State (|00> + |11>)/sqrt(2)
    # This circuit produces the |Φ⁺> Bell state. Ideal measurements yield only '00' or '11'.
    allowed_outcomes = {'00', '11'}
    actual_outcomes = set(counts.keys())

    logging.info(f"Expected outcomes (ideally): {allowed_outcomes}")
    logging.info(f"Actual outcomes observed: {actual_outcomes}")

    # Assert that only the allowed outcomes are observed (ideal simulator)
    unexpected_outcomes = actual_outcomes - allowed_outcomes
    assert not unexpected_outcomes, \
        f"Found unexpected outcomes: {unexpected_outcomes}. Only {allowed_outcomes} are expected for this Bell state circuit."

    # Assert that both expected outcomes are present (for sufficient shots)
    assert '00' in actual_outcomes, "Expected outcome '00' not found in simulation results."
    assert '11' in actual_outcomes, "Expected outcome '11' not found in simulation results."

    # Optional: Check if probabilities are roughly equal (allow some tolerance for simulation variance)
    # This is a stronger check, might fail occasionally even on ideal simulators with finite shots.
    prob_00 = counts.get('00', 0) / shots
    prob_11 = counts.get('11', 0) / shots
    logging.info(f"Observed Probabilities: P('00')={prob_00:.4f}, P('11')={prob_11:.4f}")
    # Expect probabilities around 0.5 for each. Use a reasonable tolerance, e.g., +/- 0.05 for 4096 shots.
    tolerance = 0.05
    assert abs(prob_00 - 0.5) < tolerance, f"Probability of '00' ({prob_00:.4f}) is outside the expected range [0.45, 0.55]."
    assert abs(prob_11 - 0.5) < tolerance, f"Probability of '11' ({prob_11:.4f}) is outside the expected range [0.45, 0.55]."

    logging.info("test_simulation_results passed.")

# Example of using the alternative loading strategy (from string) as a separate test
# This demonstrates the alternative requested, though file loading is preferred.
@pytest.fixture(scope="module")
def quantum_circuit_from_string() -> QuantumCircuit:
    """Loads the QuantumCircuit from an embedded QASM string."""
    try:
        logging.info("Loading QuantumCircuit from embedded QASM string as an alternative.")
        circuit = QuantumCircuit.from_qasm_str(QASM_CONTENT)
        logging.info("QuantumCircuit loaded successfully from string.")
        return circuit
    except QASM2ImportError as e:
        pytest.fail(f"Failed to parse QASM from string: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred while loading QASM from string: {e}")

def test_circuit_structure_from_string(quantum_circuit_from_string: QuantumCircuit):
    """
    Tests the basic structure of the circuit loaded directly from a string.
    """
    logging.info("Running test_circuit_structure_from_string...")
    # Re-run structural checks on the circuit loaded from the string
    assert quantum_circuit_from_string is not None
    assert quantum_circuit_from_string.num_qubits == 2
    assert quantum_circuit_from_string.num_clbits == 2
    gate_types = {instruction.operation.name for instruction in quantum_circuit_from_string.data}
    assert "h" in gate_types
    assert "cx" in gate_types
    assert "measure" in gate_types
    assert quantum_circuit_from_string.depth() > 0
    logging.info("test_circuit_structure_from_string passed.")

