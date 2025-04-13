import pytest
import logging
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.result import Counts
import os

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the expected relative path and filename for the QASM file
# Assumes this test script is in 'tests/generated/' and QASM is in 'ir/openqasm/mitigated/'
# relative to the project root.
QASM_FILENAME = "shor_example_8q_4c.qasm" # Using a descriptive name based on analysis
EXPECTED_QASM_REL_PATH_PARTS = ("ir", "openqasm", "mitigated", QASM_FILENAME)

# --- Fixtures ---

@pytest.fixture(scope="module")
def qasm_file_path() -> Path:
    """
    Provides the absolute path to the QASM file using relative path calculation.
    Handles potential FileNotFoundError.
    """
    try:
        current_file_dir = Path(__file__).parent
        # Navigate up from 'tests/generated/' to project root
        project_root = current_file_dir.parent.parent
        qasm_path = project_root.joinpath(*EXPECTED_QASM_REL_PATH_PARTS)
        logger.info(f"Calculated QASM file path: {qasm_path}")

        if not qasm_path.is_file():
            # Log an error and fail the fixture setup if file not found
            error_msg = f"QASM file not found at the expected path: {qasm_path}. " \
                        f"Ensure the file '{QASM_FILENAME}' exists at " \
                        f"'<project_root>/{'/'.join(EXPECTED_QASM_REL_PATH_PARTS)}' " \
                        f"and the test script location allows this relative path calculation."
            logger.error(error_msg)
            # Use pytest.fail to stop test execution if file is essential
            pytest.fail(error_msg)

        return qasm_path
    except Exception as e:
        # Catch any other unexpected errors during path calculation
        logger.error(f"Error calculating QASM file path: {e}", exc_info=True)
        pytest.fail(f"Fixture qasm_file_path setup failed: {e}")


@pytest.fixture(scope="module")
def quantum_circuit(qasm_file_path: Path) -> QuantumCircuit:
    """
    Loads the QuantumCircuit from the QASM file specified by the qasm_file_path fixture.
    """
    try:
        circuit = QuantumCircuit.from_qasm_file(str(qasm_file_path))
        logger.info(f"Successfully loaded QuantumCircuit from {qasm_file_path}")
        return circuit
    except Exception as e:
        logger.error(f"Failed to load QuantumCircuit from QASM file {qasm_file_path}: {e}", exc_info=True)
        # Fail the test run if the circuit cannot be loaded
        pytest.fail(f"Failed to load QuantumCircuit from {qasm_file_path}: {e}")


@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Provides an instance of the AerSimulator for running simulations."""
    logger.info("Initializing AerSimulator.")
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure of the loaded quantum circuit (qubit/clbit counts, gate types).
    """
    logger.info("--- Starting test_circuit_structure ---")
    assert quantum_circuit is not None, "Quantum circuit object should not be None"

    # Verify qubit count
    expected_qubits = 8
    actual_qubits = quantum_circuit.num_qubits
    logger.info(f"Checking qubit count: Expected={expected_qubits}, Actual={actual_qubits}")
    assert actual_qubits == expected_qubits, f"Expected {expected_qubits} qubits, but circuit has {actual_qubits}"

    # Verify classical bit count
    expected_clbits = 4
    actual_clbits = quantum_circuit.num_clbits
    logger.info(f"Checking classical bit count: Expected={expected_clbits}, Actual={actual_clbits}")
    assert actual_clbits == expected_clbits, f"Expected {expected_clbits} classical bits, but circuit has {actual_clbits}"

    # Verify presence of expected gate types for Shor's period finding
    gate_counts = quantum_circuit.count_ops()
    logger.info(f"Circuit gate counts: {gate_counts}")
    expected_gates = {'h', 'cx', 'swap', 'cz', 'measure', 'x'} # Added 'x' based on input QASM
    for gate in expected_gates:
        assert gate in gate_counts, f"Circuit is missing expected gate type: '{gate}'"
    logger.info("Gate type check passed.")

    logger.info("--- Finished test_circuit_structure ---")


def test_circuit_simulation(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on a simulator and checks basic properties of the simulation results.
    """
    logger.info("--- Starting test_circuit_simulation ---")
    shots = 2048 # Using a moderate number of shots

    # Ensure the circuit has classical bits for measurement results
    if quantum_circuit.num_clbits == 0:
         logger.warning("Circuit has no classical bits. Skipping simulation test.")
         pytest.skip("Skipping simulation test as the circuit has no classical bits for measurement.")

    logger.info(f"Running simulation with {shots} shots...")
    try:
        # Transpile might be needed for some simulators or optimization levels, but basic run should work
        # from qiskit import transpile
        # transpiled_circuit = transpile(quantum_circuit, simulator)
        # job = simulator.run(transpiled_circuit, shots=shots)
        job = simulator.run(quantum_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(quantum_circuit)
        logger.info(f"Simulation successful. Raw counts obtained: {counts}")
    except Exception as e:
        logger.error(f"Simulation execution failed: {e}", exc_info=True)
        pytest.fail(f"Circuit simulation failed: {e}")

    # Basic assertions on the counts dictionary
    assert counts is not None, "Simulation result should include counts."
    assert isinstance(counts, dict), "Counts should be a dictionary."
    assert len(counts) > 0, "Counts dictionary should not be empty after simulation."

    # Verify total counts match shots
    total_counts = sum(counts.values())
    logger.info(f"Total counts from simulation: {total_counts}, Expected shots: {shots}")
    assert total_counts == shots, f"Total counts ({total_counts}) must equal the number of shots ({shots})."

    # Verify the format of the measurement outcome bitstrings
    expected_bitstring_length = quantum_circuit.num_clbits
    logger.info(f"Verifying format of count keys (expected length {expected_bitstring_length})...")
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' is not a string."
        assert len(bitstring) == expected_bitstring_length, \
            f"Bitstring key '{bitstring}' has length {len(bitstring)}, expected {expected_bitstring_length}."
        assert all(c in '01' for c in bitstring), \
            f"Bitstring key '{bitstring}' contains invalid characters (must be 0 or 1)."
    logger.info("Count key format verification passed.")

    logger.info("--- Finished test_circuit_simulation ---")


def test_shor_period_finding_outcome(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Tests if the simulation results are consistent with Shor's period finding
    for a likely period r=4 on n=4 measurement qubits.
    Expected measurement peaks (representing s*2^n/r = s*16/4 = s*4): '0000', '0100', '1000', '1100'.
    """
    logger.info("--- Starting test_shor_period_finding_outcome ---")
    shots = 4096 # Use a higher number of shots for better statistics

    # Check if the circuit structure matches assumptions for this test
    if quantum_circuit.num_clbits != 4:
         logger.warning(f"Circuit has {quantum_circuit.num_clbits} classical bits, but this test expects 4 for Shor's period finding (r=4, n=4). Skipping.")
         pytest.skip("Skipping Shor's outcome test as it requires 4 classical bits.")

    logger.info(f"Running simulation for Shor's outcome analysis with {shots} shots...")
    try:
        job = simulator.run(quantum_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(quantum_circuit)
        logger.info(f"Simulation counts for Shor's test: {counts}")
    except Exception as e:
        logger.error(f"Simulation execution failed during Shor's test: {e}", exc_info=True)
        pytest.fail(f"Circuit simulation failed during Shor's test: {e}")

    assert counts is not None and len(counts) > 0, "Simulation should produce non-empty counts for Shor's test."

    # Define the expected outcomes for period r=4 on n=4 measurement qubits (s*2^n/r = s*4)
    # s=0 -> 0 ('0000'), s=1 -> 4 ('0100'), s=2 -> 8 ('1000'), s=3 -> 12 ('1100')
    expected_period_finding_peaks = {'0000', '0100', '1000', '1100'}
    logger.info(f"Expected primary measurement outcomes (peaks) for period r=4: {expected_period_finding_peaks}")

    # Calculate the total probability mass concentrated on the expected peaks
    probability_on_expected = sum(counts.get(outcome, 0) for outcome in expected_period_finding_peaks) / shots
    logger.info(f"Probability mass on expected outcomes {expected_period_finding_peaks}: {probability_on_expected:.4f}")

    # Assert that a significant portion of the probability distribution falls on the expected outcomes.
    # This threshold accounts for potential noise or imperfections in a real Shor implementation simulation.
    # A perfect IQFT would yield only these outcomes (with equal probability ideally).
    # A threshold significantly > 0.5 indicates the period finding likely succeeded.
    min_expected_probability_threshold = 0.6 # Adjusted threshold, can be tuned
    assert probability_on_expected >= min_expected_probability_threshold, \
        f"The probability mass on the expected period-finding peaks ({probability_on_expected:.4f}) " \
        f"is below the threshold ({min_expected_probability_threshold}). " \
        f"Expected peaks {expected_period_finding_peaks} were not sufficiently dominant."

    # Optional: Check if the most frequent results are among the expected ones
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_n = min(len(sorted_counts), 4) # Look at the top 4 peaks
    most_frequent_outcomes = {item[0] for item in sorted_counts[:top_n]}
    logger.info(f"Top {top_n} most frequent outcomes: {most_frequent_outcomes}")
    # Check if the most frequent are exactly the expected ones (can be strict)
    # assert most_frequent_outcomes == expected_period_finding_peaks, \
    #    f"The set of most frequent outcomes {most_frequent_outcomes} does not exactly match the expected peaks {expected_period_finding_peaks}"

    logger.info("Result distribution is consistent with expected Shor's period finding (r=4) outcomes.")
    logger.info("--- Finished test_shor_period_finding_outcome ---")

