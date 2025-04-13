import pytest
import logging
import math
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.result import Counts
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Test Configuration ---
QASM_FILENAME = "shor_period_finding_8q_simulated.qasm" # Placeholder filename if loading from file
SHOTS = 4096

# --- QASM Content ---
# Using the provided QASM content directly as a string
qasm_content = """
OPENQASM 2.0;
include "qelib1.inc";

qreg period[4];
qreg target[4];
creg reg_measure[4];

x target[0];
h period[0];
h period[1];
h period[2];
h period[3];
cx period[3],target[0];
cx period[3],target[1];
cx period[3],target[2];
cx period[2],target[2];
swap period[0],period[3];
swap period[1],period[2];
h period[0];
h period[1];
h period[2];
h period[3];
cz period[0],period[1];
cz period[0],period[2];
cz period[1],period[2];
cz period[0],period[3];
cz period[1],period[3];
cz period[2],period[3];
measure period[0] -> reg_measure[0];
measure period[1] -> reg_measure[1];
measure period[2] -> reg_measure[2];
measure period[3] -> reg_measure[3];
"""

# --- Fixtures ---

@pytest.fixture(scope="module")
def quantum_circuit() -> QuantumCircuit:
    """Fixture to load the QuantumCircuit from the QASM string."""
    try:
        circuit = QuantumCircuit.from_qasm_str(qasm_content)
        logging.info(f"Successfully loaded QuantumCircuit from QASM string.")
        return circuit
    except Exception as e:
        logging.error(f"Failed to load QuantumCircuit from QASM string: {e}")
        pytest.fail(f"Failed to load QuantumCircuit from QASM string: {e}")

@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """Fixture to provide an AerSimulator instance."""
    logging.info("Initializing AerSimulator.")
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """Tests the basic structural properties of the loaded quantum circuit."""
    logging.info("Running test_circuit_structure...")
    assert quantum_circuit is not None, "Quantum circuit object should not be None"

    expected_qubits = 8
    expected_clbits = 4
    logging.info(f"Checking circuit dimensions: qubits={quantum_circuit.num_qubits}, clbits={quantum_circuit.num_clbits}")
    assert quantum_circuit.num_qubits == expected_qubits, f"Circuit should have {expected_qubits} qubits"
    assert quantum_circuit.num_clbits == expected_clbits, f"Circuit should have {expected_clbits} classical bits"

    # Check for presence of key gate types
    gate_counts = quantum_circuit.count_ops()
    logging.info(f"Circuit gate counts: {gate_counts}")
    assert "measure" in gate_counts, "Circuit should contain measurement operations"
    assert gate_counts.get("measure", 0) == expected_clbits, f"Should measure all {expected_clbits} classical bits"
    assert "h" in gate_counts, "Circuit should contain Hadamard gates"
    assert "cx" in gate_counts or "CX" in gate_counts, "Circuit should contain CNOT (cx) gates"
    assert "cz" in gate_counts or "CZ" in gate_counts, "Circuit should contain CZ gates"
    assert "swap" in gate_counts, "Circuit should contain SWAP gates"

    logging.info("test_circuit_structure finished successfully.")


def test_circuit_simulation(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """Runs the circuit on the AerSimulator and performs basic checks on results."""
    logging.info(f"Running test_circuit_simulation with {SHOTS} shots...")
    assert quantum_circuit is not None, "Quantum circuit fixture failed"
    assert simulator is not None, "Simulator fixture failed"
    assert quantum_circuit.num_clbits > 0, "Circuit must have classical bits for measurement results"

    # Run the simulation
    job = simulator.run(quantum_circuit, shots=SHOTS)
    result = job.result()
    counts = result.get_counts(quantum_circuit)
    logging.info(f"Simulation finished. Result counts: {counts}")

    # Basic assertions on results
    assert counts is not None, "Simulation should produce counts"
    assert isinstance(counts, dict), "Counts should be a dictionary"
    assert len(counts) > 0, "Counts dictionary should not be empty"

    # Verify total counts match shots
    total_counts = sum(counts.values())
    logging.info(f"Total counts: {total_counts}, Expected shots: {SHOTS}")
    assert total_counts == SHOTS, f"Total counts ({total_counts}) should equal the number of shots ({SHOTS})"

    # Verify format of result keys (bitstrings)
    expected_bitstring_length = quantum_circuit.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), "Result keys should be strings"
        assert len(bitstring) == expected_bitstring_length, \
            f"Result bitstring '{bitstring}' length should match number of classical bits ({expected_bitstring_length})"
        assert all(c in '01' for c in bitstring), \
            f"Result bitstring '{bitstring}' should only contain '0' or '1'"

    logging.info("test_circuit_simulation finished successfully.")


def test_shor_period_finding_outcome_distribution(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Checks if the simulation results exhibit characteristics expected from a
    period-finding algorithm (non-uniform distribution with distinct peaks).
    """
    logging.info(f"Running test_shor_period_finding_outcome_distribution with {SHOTS} shots...")
    assert quantum_circuit is not None, "Quantum circuit fixture failed"
    assert simulator is not None, "Simulator fixture failed"
    num_measurement_qubits = quantum_circuit.num_clbits
    assert num_measurement_qubits == 4, "This test assumes 4 measurement qubits based on the circuit structure"

    # Run the simulation
    job = simulator.run(quantum_circuit, shots=SHOTS)
    result = job.result()
    counts = result.get_counts(quantum_circuit)
    logging.info(f"Simulation finished for distribution analysis. Result counts: {counts}")

    assert len(counts) > 0, "Counts dictionary should not be empty"

    # Check for non-uniformity: A successful period-finding run should have peaks.
    # A perfectly uniform distribution over 2^n states would give each state SHOTS / (2^n) counts.
    num_possible_states = 2**num_measurement_qubits
    ideal_uniform_prob = 1.0 / num_possible_states
    max_prob = max(c / SHOTS for c in counts.values())

    # Assert that the maximum probability is significantly higher than uniform
    # Threshold chosen heuristically - might need adjustment for different circuits/shots
    significance_threshold = 1.5
    logging.info(f"Max observed probability: {max_prob:.4f}, Ideal uniform probability: {ideal_uniform_prob:.4f}")
    assert max_prob > significance_threshold * ideal_uniform_prob, \
        f"Distribution seems too uniform. Max probability ({max_prob:.4f}) is not significantly larger than uniform ({ideal_uniform_prob:.4f})."

    # Check if the most probable states account for a large portion of the counts.
    # For period finding, we expect a few peaks corresponding to k*N/r.
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_n = min(4, len(sorted_counts)) # Look at top 4 peaks or fewer if less results
    top_counts_sum = sum(count for state, count in sorted_counts[:top_n])
    fraction_in_top = top_counts_sum / SHOTS

    # Expect the top peaks to capture a significant fraction of the total probability.
    # Threshold chosen heuristically.
    min_fraction_threshold = 0.5
    logging.info(f"Fraction of counts in top {top_n} states: {fraction_in_top:.4f}")
    assert fraction_in_top >= min_fraction_threshold, \
        f"Top {top_n} states account for only {fraction_in_top:.4f} of counts, expected >= {min_fraction_threshold}. Peaks might not be distinct enough."

    # Check if specific expected peaks for period r=4 (0000, 0100, 1000, 1100) are present and prominent.
    # This is based on the visual inspection of the circuit suggesting period finding, potentially for N=15 (a=7 or 11 -> r=4).
    expected_peaks = {'0000', '0100', '1000', '1100'} # Corresponds to 0, 4, 8, 12
    observed_peaks = {state for state, count in sorted_counts[:top_n]}
    
    # Check if at least some expected peaks are among the most observed ones
    common_peaks = expected_peaks.intersection(observed_peaks)
    logging.info(f"Expected prominent peaks (for r=4): {expected_peaks}")
    logging.info(f"Observed top {top_n} peaks: {observed_peaks}")
    logging.info(f"Common peaks between expected and observed top peaks: {common_peaks}")
    
    # We expect at least some overlap, ideally most of the probability mass is in expected peaks
    # Allow for some noise / deviation
    assert len(common_peaks) > 0, "None of the expected peaks (0000, 0100, 1000, 1100) are among the most probable results."
    
    # More stringent check: ensure the expected peaks capture a large fraction of the probability
    expected_peaks_counts_sum = sum(counts.get(peak, 0) for peak in expected_peaks)
    expected_peaks_fraction = expected_peaks_counts_sum / SHOTS
    logging.info(f"Fraction of counts in expected peaks {expected_peaks}: {expected_peaks_fraction:.4f}")
    assert expected_peaks_fraction >= min_fraction_threshold, \
         f"Expected peaks {expected_peaks} account for only {expected_peaks_fraction:.4f} of counts, expected >= {min_fraction_threshold}."


    logging.info("test_shor_period_finding_outcome_distribution finished successfully.")

# --- Helper Functions (Optional) ---
# Example: A function to calculate continued fractions could be added here
# if needed for more advanced Shor's algorithm post-processing tests.

