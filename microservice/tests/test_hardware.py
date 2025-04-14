#!/usr/bin/env python3
"""
Test script for all supported quantum hardware providers.
"""
import requests
import json
import time
from typing import Dict, Any

API_URL = "http://localhost:8889/api/v1/circuits/execute"

def test_hardware(provider: str) -> Dict[str, Any]:
    """
    Test a specific quantum hardware.
    
    Args:
        provider: The hardware provider (ibm)
        
    Returns:
        API response
    """
    print(f"\n===== Testing {provider} hardware =====")
    
    # Read the QASM file
    #with open("../my_circuit.qasm", "r") as f:
    #    circuit = f.read()
    
    # Create the payload
    payload = {
        #"circuit": circuit,
        "shots": 1024,
        "backend_type": "hardware",
        "backend_provider": provider,
        "backend_name": "ibmq_qasm_simulator",
        "async_mode": False
    }
    
    # Make the API call
    print(f"Sending request to {provider} hardware...")
    response = requests.post(
        API_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    # Print response
    result = response.json()
    print(json.dumps(result, indent=2))
    
    print(f"===== {provider} test complete =====\n")
    return result

def main():
    """Test all supported hardware providers."""
    # Test each supported provider
    providers = ["ibm"]
    
    for provider in providers:
        result = test_hardware(provider)
        # Add a small delay between requests
        time.sleep(1)
    
    print("All hardware tests completed!")
    print("Note: To use IBM hardware, you must set the IBM_QUANTUM_TOKEN environment variable.")
    print("docker run -p 8889:8889 -e IBM_QUANTUM_TOKEN=your_token_here your-image-name")

if __name__ == "__main__":
    main()
