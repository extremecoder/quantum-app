#!/usr/bin/env python3
"""
API Test Script for Quantum Microservice.
"""
import argparse
import json
import requests
import time
import os
from pathlib import Path

class QuantumAPIClient:
    """Client for interacting with the Quantum Microservice API."""
    
    def __init__(self, host: str = "localhost", port: int = 8889, api_base: str = "/api/v1"):
        """
        Initialize API client.
        
        Args:
            host: API host
            port: API port
            api_base: Base API path
        """
        self.base_url = f"http://{host}:{port}{api_base}"
        print(f"API URL: {self.base_url}")
    
    def execute_circuit(self, circuit_path: str, shots: int = 1024, 
                       backend_type: str = "simulator", backend_provider: str = "qiskit",
                       backend_name: str = None, async_mode: bool = False,
                       parameters: dict = None) -> dict:
        """
        Execute a quantum circuit.
        
        Args:
            circuit_path: Path to OpenQASM circuit file
            shots: Number of execution shots
            backend_type: "simulator" or "hardware"
            backend_provider: Provider name
            backend_name: Specific backend name (optional)
            async_mode: Whether to run in async mode
            parameters: Circuit parameters (optional)
            
        Returns:
            API response as dict
        """
        # Read circuit file
        with open(circuit_path, 'r') as f:
            circuit_content = f.read()
        
        # Build request payload
        payload = {
            "circuit": circuit_content,
            "shots": shots,
            "backend_type": backend_type,
            "backend_provider": backend_provider,
            "async_mode": async_mode
        }
        
        # Add optional parameters
        if backend_name:
            payload["backend_name"] = backend_name
        if parameters:
            payload["parameters"] = parameters
        
        # Execute API call
        response = requests.post(
            f"{self.base_url}/circuits/execute",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Handle response
        if response.status_code in [200, 202]:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {"status": "error", "error": response.text}
    
    def get_job_status(self, job_id: str) -> dict:
        """
        Get status of a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            API response as dict
        """
        response = requests.get(f"{self.base_url}/jobs/{job_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {"status": "error", "error": response.text}
    
    def cancel_job(self, job_id: str) -> dict:
        """
        Cancel a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            API response as dict
        """
        response = requests.delete(f"{self.base_url}/jobs/{job_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {"status": "error", "error": response.text}

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Quantum Microservice API")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8889, help="API port")
    parser.add_argument("--circuit", default="../my_circuit.qasm", help="Path to QASM circuit file")
    parser.add_argument("--test", choices=["sync", "async", "all"], default="all", 
                        help="Test mode: sync, async, or all")
    
    args = parser.parse_args()
    
    # Initialize API client
    client = QuantumAPIClient(host=args.host, port=args.port)
    
    circuit_path = args.circuit
    if not os.path.isabs(circuit_path):
        circuit_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), circuit_path)
    
    if not os.path.exists(circuit_path):
        print(f"Error: Circuit file not found: {circuit_path}")
        return
    
    print(f"Testing with circuit: {circuit_path}")
    
    # Test synchronous execution
    if args.test in ["sync", "all"]:
        print("\n=== Testing Synchronous Circuit Execution ===")
        sync_result = client.execute_circuit(
            circuit_path=circuit_path,
            shots=1024,
            backend_type="simulator",
            backend_provider="qiskit",
            async_mode=False
        )
        print("Synchronous execution result:")
        print(json.dumps(sync_result, indent=2))
    
    # Test asynchronous execution
    if args.test in ["async", "all"]:
        print("\n=== Testing Asynchronous Circuit Execution ===")
        async_result = client.execute_circuit(
            circuit_path=circuit_path,
            shots=1024,
            backend_type="simulator",
            backend_provider="qiskit",
            async_mode=True
        )
        print("Asynchronous execution result:")
        print(json.dumps(async_result, indent=2))
        
        if async_result.get("status") == "success" and "job_id" in async_result.get("data", {}):
            job_id = async_result["data"]["job_id"]
            
            # Test job status
            print("\n=== Testing Job Status ===")
            time.sleep(1)  # Wait a bit for job to process
            status_result = client.get_job_status(job_id)
            print("Job status result:")
            print(json.dumps(status_result, indent=2))
            
            # Wait and check again if not completed
            if status_result.get("data", {}).get("status") not in ["COMPLETED", "FAILED"]:
                print("Waiting for job to complete...")
                time.sleep(3)
                status_result = client.get_job_status(job_id)
                print("Updated job status:")
                print(json.dumps(status_result, indent=2))
            
            # Test job cancellation if still running
            if status_result.get("data", {}).get("status") in ["QUEUED", "RUNNING"]:
                print("\n=== Testing Job Cancellation ===")
                cancel_result = client.cancel_job(job_id)
                print("Job cancellation result:")
                print(json.dumps(cancel_result, indent=2))
    
    print("\n=== API Testing Complete ===")

if __name__ == "__main__":
    main()
