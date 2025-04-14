import pytest
import logging
import sys
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- QASM Content ---
# Using the embedded QASM string method as requested in the alternative strategy
# for guaranteed runnability without file path dependencies.
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

# --- Pytest Fixtures ---

@pytest.fixture(scope="module")
def quantum_circuit() -> QuantumCircuit:
    """
    Loads the QuantumCircuit from the embedded QASM string.
    """
    try:
        circuit = QuantumCircuit.from_qasm_str(QASM_CONTENT)
        logging.info("Successfully loaded QuantumCircuit from QASM string.")
        return circuit
    except Exception as e:
        logging.error(f"Failed to load QuantumCircuit from QASM string: {e}")
        pytest.fail(f"Failed to load QuantumCircuit: {e}")

@pytest.fixture(scope="module")
def simulator() -> AerSimulator:
    """
    Provides an instance of the AerSimulator.
    """
    logging.info("Initializing AerSimulator.")
    return AerSimulator()

# --- Test Functions ---

def test_circuit_structure(quantum_circuit: QuantumCircuit):
    """
    Tests the basic structure of the loaded quantum circuit.
    """
    logging.info("Running test: test_circuit_structure")
    assert quantum_circuit is not None, "QuantumCircuit object should not be None"

    # Check qubit and classical bit counts
    expected_qubits = 2
    expected_clbits = 2
    assert quantum_circuit.num_qubits == expected_qubits, f"Expected {expected_qubits} qubits, but got {quantum_circuit.num_qubits}"
    assert quantum_circuit.num_clbits == expected_clbits, f"Expected {expected_clbits} classical bits, but got {quantum_circuit.num_clbits}"
    logging.info(f"Circuit has {quantum_circuit.num_qubits} qubits and {quantum_circuit.num_clbits} classical bits.")

    # Check for presence of expected gate types
    operations = {instr.operation.name for instr in quantum_circuit.data}
    expected_ops = {'h', 'cx', 'measure'}
    assert expected_ops.issubset(operations), f"Expected operations {expected_ops} not found in circuit operations {operations}"
    logging.info(f"Found operations: {operations}")

    # Log circuit depth
    logging.info(f"Circuit depth: {quantum_circuit.depth()}")


def test_circuit_simulation(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Runs the circuit on the AerSimulator and performs basic checks on the results.
    """
    logging.info("Running test: test_circuit_simulation")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed"
    assert simulator is not None, "Simulator fixture failed"

    shots = 1024  # Reasonable number of shots for statistical relevance

    # Transpile for the simulator (optional but good practice)
    try:
        transpiled_circuit = transpile(quantum_circuit, simulator)
    except Exception as e:
        pytest.fail(f"Circuit transpilation failed: {e}")

    # Run the simulation
    logging.info(f"Running simulation with {shots} shots...")
    try:
        result = simulator.run(transpiled_circuit, shots=shots).result()
        counts = result.get_counts(transpiled_circuit)
        logging.info(f"Simulation successful. Counts: {counts}")
    except Exception as e:
        pytest.fail(f"Simulation failed: {e}")

    # Basic assertions on results
    assert counts is not None, "Simulation result should include counts"
    assert isinstance(counts, dict), "Counts should be a dictionary"
    assert sum(counts.values()) == shots, f"Total counts ({sum(counts.values())}) should equal the number of shots ({shots})"

    # Check format of result keys (bitstrings)
    num_clbits = quantum_circuit.num_clbits
    for bitstring in counts.keys():
        assert isinstance(bitstring, str), f"Count key '{bitstring}' is not a string"
        # Qiskit counts keys are typically reversed (c[1]c[0]) compared to QASM definition (c[0]c[1])
        # Here we just check length and content
        assert len(bitstring) == num_clbits, f"Bitstring '{bitstring}' has incorrect length. Expected {num_clbits}."
        assert all(c in '01' for c in bitstring), f"Bitstring '{bitstring}' contains invalid characters."

    logging.info("Basic simulation result checks passed.")


def test_bell_state_properties(quantum_circuit: QuantumCircuit, simulator: AerSimulator):
    """
    Tests the expected outcomes for a Bell state (|Φ+>) preparation circuit.
    Ideally, only '00' and '11' states should be observed.
    """
    logging.info("Running test: test_bell_state_properties")
    assert quantum_circuit is not None, "QuantumCircuit fixture failed"
    assert simulator is not None, "Simulator fixture failed"

    shots = 4096  # Use more shots for better statistics on specific states

    # Transpile and run
    try:
        transpiled_circuit = transpile(quantum_circuit, simulator)
        result = simulator.run(transpiled_circuit, shots=shots).result()
        counts = result.get_counts(transpiled_circuit)
        logging.info(f"Bell state simulation counts ({shots} shots): {counts}")
    except Exception as e:
        pytest.fail(f"Bell state simulation failed: {e}")

    # Assertions for Bell state |Φ+> = (|00> + |11>)/sqrt(2)
    # In an ideal simulation, only '00' and '11' outcomes should appear.
    # Note: Qiskit's counts dictionary keys represent the classical register state c[n-1]...c[0].
    # For creg c[2] (c[0], c[1]), the states are measured q[0]->c[0], q[1]->c[1].
    # The key '00' corresponds to c[1]=0, c[0]=0.
    # The key '11' corresponds to c[1]=1, c[0]=1.
    # These directly match the expected entangled states |00> and |11>.
    expected_states = {'00', '11'}
    observed_states = set(counts.keys())

    # Check that only the expected states are present
    assert observed_states.issubset(expected_states), \
        f"Observed states {observed_states} contain unexpected outcomes. Expected only {expected_states}."

    # Check that both expected states are present (might fail with very low shots or noise)
    if shots > 50: # Avoid failing if shots are too low to guarantee both outcomes
        assert expected_states.issubset(observed_states), \
            f"Expected states {expected_states} were not all observed. Got {observed_states}."

    # Optional: Check if probabilities are roughly equal (within some tolerance)
    # This is more prone to statistical fluctuations.
    # count_00 = counts.get('00', 0)
    # count_11 = counts.get('11', 0)
    # assert abs(count_00 - count_11) / shots < 0.1, \
    #     f"Counts for '00' ({count_00}) and '11' ({count_11}) are not approximately equal for {shots} shots."

    logging.info("Bell state property checks passed: Observed outcomes match expected entangled states.")

# --- Main execution block for running pytest ---
# This allows running the script directly using `python <filename>.py`
# although the standard way is `pytest <filename>.py`
if __name__ == "__main__":
    # You can add specific pytest commands here if needed,
    # but typically pytest is run from the command line.
    # Example: pytest.main(['-v', __file__])
    print("To run these tests, navigate to the directory containing this file and run:")
    print(f"pytest {Path(__file__).name}")
