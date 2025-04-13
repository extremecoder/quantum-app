# my-cli-test-app/source/circuits/bell_qiskit.py
"""
A simple Qiskit circuit for generating a Bell state |Φ+>.
"""

# Ensure Qiskit is installed: pip install qiskit
try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
except ImportError:
    print("Qiskit not found. Please install it: pip install qiskit")
    # Or handle the error appropriately if this script is part of a larger system
    raise

def create_circuit():
    """Creates and returns a Bell state circuit."""
    # Create a quantum circuit with 2 qubits and 2 classical bits
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(qr, cr)

    # Apply Hadamard gate to the first qubit
    circuit.h(qr[0])

    # Apply CNOT gate with control qubit 0 and target qubit 1
    circuit.cx(qr[0], qr[1])

    # Map the quantum measurement to the classical bits
    circuit.measure(qr, cr)

    return circuit

# Optional: If you want to define the circuit directly as an object
# qc = create_circuit()

if __name__ == '__main__':
    # Example of how to use the function
    bell_circuit = create_circuit()
    print("Bell state circuit created:")
    print(bell_circuit)
    # You can also export QASM directly here for testing
    # print("\nQASM Output:")
    # print(bell_circuit.qasm()) 