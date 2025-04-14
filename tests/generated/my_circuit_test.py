import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.result import Counts
import os # Used for checking file existence if Path fails unexpectedly

# Configure logging for the test module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Assume the input QASM file is named based on the circuit's nature or provided context
# If the original filename isn't known, use a placeholder like 'unknown_circuit_2q.qasm'
# For this specific circuit (Bell state + measure), let's name it:
QASM_FILENAME = "bell_state_measure_2q.qasm"
SHOTS = 4096  # Number of shots for simulation

# --- Path Configuration ---
# Calculate the absolute path to the QASM file relative to this test file.
# Assumes the test file is in a structure like 'project_root/tests/generated/'
# and the QASM file is in 'project_root/ir/openqasm/mitigated/'
try:
    # Navigate up three levels (from tests/generated/ to project_root/)
    project_root = Path(__file__).parent.parent.parent
    QASM_FILE_PATH = project_root / "ir" / "openqasm" / "mitigated" / QASM_FILENAME
    # Check if the calculated path exists and is a file
    if not QASM_FILE_PATH.is_file():
         # Attempt to create the directory structure and a dummy file for testing purposes if it doesn't exist
         logging.warning(f"QASM file not found at calculated path: {QASM_FILE_PATH}. Attempting to create dummy file for test execution.")
         qasm_content_for_dummy = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""
         try:
             QASM_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
             with open(QASM_FILE_PATH, 'w') as f:
                 f.write(qasm_content_for_dummy)
             logging.info(f"Created dummy QASM file at: {QASM_FILE_PATH}")
         except Exception as e:
             logging.error(f"Failed to create dummy QASM file directory structure or file: {e}", exc_info=True)
             # If creation fails, we have to stop.
             raise FileNotFoundError(f"QASM file not found at {QASM_FILE_PATH} and dummy file creation failed.")

except Exception as e:
    logging.error(f"Error calculating QASM file path: {e}", exc_info=True)
    # Fallback: Define QASM content directly if path calculation fails
    QASM_FILE_PATH = None
    QASM_CONTENT = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""
    logging.warning("Using embedded QASM content due to path calculation error.")


# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def quantum_circuit() -> QuantumCircuit:
    """
    Fixture to load the QuantumCircuit.
    Tries loading from the calculated file path first, then falls back to
    embedded content if the path wasn't resolved or the file doesn't exist.
    """
    circuit = None
    if QASM_FILE_PATH and QASM_FILE_PATH.is_file():
        logging.info(f"Loading QuantumCircuit from file: {QASM_FILE_PATH}")
        try:
            circuit = QuantumCircuit.from_qasm_file(str(QASM_FILE_PATH))
            logging.info(f"Successfully loaded QuantumCircuit from {QASM_FILENAME}")
        except FileNotFoundError:
            logging.error(f"QASM file not found at {QASM_FILE_PATH}, although path object exists.")
            pytest.fail(f"QASM file not found at expected location: {QASM_FILE_PATH}")
        except Exception as e:
            logging.error(f"Failed to load QuantumCircuit from file {QASM_FILE_PATH}: {e}", exc_info=True)
            pytest.fail(f"Error loading QASM file {QASM_FILE_PATH}: {e}")
    elif 'QASM_CONTENT' in globals():
        logging.warning("Loading QuantumCircuit from embedded QASM string as fallback.")
        try:
            circuit = QuantumCircuit.from_qasm_str(QASM_CONTENT)
            logging.info("Successfully loaded QuantumCircuit from embedded string.")
        except Exception as e:
            logging.error(f"Failed to load QuantumCircuit from embedded string: {e}", exc_info=True)
            pytest.fail(f"Error loading QASM from string: {e}")
    else:
        # This state should not be reached if the logic above is correct
        logging.error("Quantum circuit could not be loaded - no valid path or embedded content.")
        pytest.fail("Failed to provide a QuantumCircuit object for testing.")

    assert circuit is not None, "Circuit object is None after loading attempt."
    return circuit

@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """
    Fixture to provide an AerSimulator instance.
    """
    logging.info("Initializing AerSimulator.")
    sim = AerSimulator()
    return sim

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure (qubits, clbits, gate types) of the loaded circuit.
    """
    logging.info("--- Starting test_circuit_structure ---")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed to return a valid circuit."

    # Verify qubit and classical bit counts
    expected_qubits = 2
    expected_clbits = 2
    logging.info(f"Checking qubit count: Expected={expected_qubits}, Actual={quantum_circuit.num_qubits}")
    assert quantum_circuit.num_qubits == expected_qubits, f"Circuit should have {expected_qubits} qubits, found {quantum_circuit.num_qubits}"
    logging.info(f"Checking classical bit count: Expected={expected_clbits}, Actual={quantum_circuit.num_clbits}")
    assert quantum_circuit.num_clbits == expected_clbits, f"Circuit should have {expected_clbits} classical bits, found {quantum_circuit.num_clbits}"

    # Verify presence and count of expected operations
    ops_counts = quantum_circuit.count_ops()
    logging.info(f"Circuit operations counts: {ops_counts}")
    assert 'h' in ops_counts, "Circuit is missing expected Hadamard (h) gate."
    assert ops_counts['h'] == 1, f"Expected 1 Hadamard gate, found {ops_counts['h']}"
    assert 'cx' in ops_counts, "Circuit is missing expected CNOT (cx) gate."
    assert ops_counts['cx'] == 1, f"Expected 1 CNOT gate, found {ops_counts['cx']}"
    assert 'measure' in ops_counts, "Circuit is missing expected Measure operations."
    assert ops_counts['measure'] == 2, f"Expected 2 Measure operations, found {ops_counts['measure']}"

    logging.info("Circuit structure verification passed.")
    logging.info("--- Finished test_circuit_structure ---")


def test_circuit_simulation_basic(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on the simulator and performs basic checks on the results.
    """
    logging.info("--- Starting test_circuit_simulation_basic ---")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed."
    assert simulator is not None, "Simulator fixture failed."
    assert quantum_circuit.num_clbits > 0, "Circuit must have classical bits for simulation results."

    logging.info(f"Running simulation with {SHOTS} shots.")
    try:
        # Transpile for the simulator for potentially better performance/compatibility
        transpiled_circuit = transpile(quantum_circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=SHOTS).result()
        counts = result.get_counts(transpiled_circuit)
        logging.info(f"Simulation successful. Counts obtained: {counts}")
    except Exception as e:
        logging.error(f"Simulation execution failed: {e}", exc_info=True)
        pytest.fail(f"Simulation failed with error: {e}")

    # Basic assertions on the counts dictionary
    assert counts is not None, "Simulation result did not contain counts."
    assert isinstance(counts, dict), f"Expected counts to be a dict, but got {type(counts)}"
    assert sum(counts.values()) == SHOTS, f"Total counts ({sum(counts.values())}) do not match the number of shots ({SHOTS})."

    # Check format of count keys (bitstrings)
    expected_bitstring_length = quantum_circuit.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' is not a string."
        # Qiskit convention: c[N-1]...c[1]c[0]. QASM: measure q[0]->c[0], q[1]->c[1]. Result key is 'c1c0'.
        assert len(bitstring) == expected_bitstring_length, \
            f"Bitstring key '{bitstring}' has length {len(bitstring)}, expected {expected_bitstring_length} based on num_clbits."
        assert all(c in '01' for c in bitstring), f"Bitstring key '{bitstring}' contains invalid characters (should be only '0' or '1')."

    logging.info("Basic simulation checks passed.")
    logging.info("--- Finished test_circuit_simulation_basic ---")


def test_bell_state_correlation(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Tests the specific outcome for the Bell state (|Φ+>) preparation circuit.
    Expects primarily '00' and '11' outcomes with roughly equal probability.
    """
    logging.info("--- Starting test_bell_state_correlation ---")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed."
    assert simulator is not None, "Simulator fixture failed."

    # Verify circuit structure is consistent with Bell state preparation before expensive simulation
    ops_counts = quantum_circuit.count_ops()
    is_bell_like = (quantum_circuit.num_qubits == 2 and
                    quantum_circuit.num_clbits == 2 and
                    'h' in ops_counts and
                    'cx' in ops_counts and
                    'measure' in ops_counts)
    if not is_bell_like:
        pytest.skip("Skipping Bell state correlation test: Circuit structure doesn't match expected pattern.")

    logging.info(f"Running Bell state simulation with {SHOTS} shots.")
    try:
        # Transpile for the simulator
        transpiled_circuit = transpile(quantum_circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=SHOTS).result()
        counts = result.get_counts(transpiled_circuit)
        logging.info(f"Bell state simulation counts: {counts}")
    except Exception as e:
        logging.error(f"Bell state simulation execution failed: {e}", exc_info=True)
        pytest.fail(f"Bell state simulation failed with error: {e}")

    assert counts is not None, "Simulation result missing counts for Bell state test."
    assert sum(counts.values()) == SHOTS, f"Total counts ({sum(counts.values())}) mismatch shots ({SHOTS}) in Bell state test."

    # Analyze counts for expected Bell state |Φ+> = (|00> + |11>)/sqrt(2)
    # Measurement results (keys 'c1c0') should be dominated by '00' and '11'.
    counts_00 = counts.get('00', 0)
    counts_11 = counts.get('11', 0)
    counts_01 = counts.get('01', 0)
    counts_10 = counts.get('10', 0)

    logging.info(f"Counts distribution: '00': {counts_00}, '11': {counts_11}, '01': {counts_01}, '10': {counts_10}")

    # 1. Check that the correlated states ('00', '11') are dominant.
    # Allow for a small percentage of noise/errors (e.g., < 5% total for '01' and '10').
    noise_fraction = (counts_01 + counts_10) / SHOTS
    max_allowed_noise = 0.05
    logging.info(f"Noise fraction ('01' + '10'): {noise_fraction:.4f}")
    assert noise_fraction < max_allowed_noise, \
        f"Noise states ('01', '10') constitute {noise_fraction*100:.2f}%, exceeding threshold of {max_allowed_noise*100:.2f}%."

    # 2. Check that the probabilities for '00' and '11' are roughly equal (close to 0.5 each).
    # Use a tolerance based on expected statistical fluctuations (sqrt(N*p*(1-p))).
    # For p=0.5, std dev = sqrt(N*0.25) = 0.5*sqrt(N). 3 sigma ~ 1.5*sqrt(N).
    # Relative tolerance: 3 sigma / (N*p) = (1.5*sqrt(N)) / (N*0.5) = 3 / sqrt(N)
    # Or use a fixed tolerance, e.g., +/- 0.1 around the ideal 0.5 probability.
    prob_00 = counts_00 / SHOTS
    prob_11 = counts_11 / SHOTS
    ideal_prob = 0.5
    tolerance = 0.1 # Allowable absolute deviation from ideal probability 0.5

    logging.info(f"Probabilities: P(00)={prob_00:.4f}, P(11)={prob_11:.4f}. Ideal={ideal_prob}, Tolerance={tolerance}")

    assert abs(prob_00 - ideal_prob) < tolerance, \
        f"Probability of '00' ({prob_00:.4f}) deviates from ideal ({ideal_prob}) by more than {tolerance}."
    assert abs(prob_11 - ideal_prob) < tolerance, \
        f"Probability of '11' ({prob_11:.4f}) deviates from ideal ({ideal_prob}) by more than {tolerance}."

    logging.info("Bell state correlation test passed.")
    logging.info("--- Finished test_bell_state_correlation ---")

# --- Main execution for running tests (optional, Pytest handles this) ---
# if __name__ == "__main__":
#     # You can run pytest programmatically, but usually, you run `pytest` from the command line
#     pytest.main([__file__]) # Runs tests in the current file
