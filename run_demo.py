#!/usr/bin/env python3
"""
Spam Classification API Demo

This script demonstrates the complete workflow:
1. Training the model
2. Starting the API server
3. Making predictions
"""

import os
import time
import requests
import subprocess
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return the result"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Success: {result.stdout}")
    return True

def train_model():
    """Train the spam classification model"""
    print("=" * 50)
    print("STEP 1: Training the Model")
    print("=" * 50)
    
    # Check if model already exists
    model_path = Path("models/spam_classifier")
    if model_path.exists():
        print("Model already exists. Skipping training.")
        return True
    
    # Run training script
    success = run_command("python scripts/train_model.py")
    if success:
        print("Model training completed successfully!")
        return True
    else:
        print("Model training failed!")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("=" * 50)
    print("STEP 2: Starting API Server")
    print("=" * 50)
    
    # Change to api directory
    os.chdir("api")
    
    # Start server in background
    server_process = subprocess.Popen(
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            print("API server started successfully!")
            return server_process
        else:
            print("API server failed to start properly")
            server_process.terminate()
            return None
    except requests.exceptions.RequestException:
        print("API server failed to start")
        server_process.terminate()
        return None

def test_api():
    """Test the API with sample predictions"""
    print("=" * 50)
    print("STEP 3: Testing API")
    print("=" * 50)
    
    # Sample test cases
    test_cases = [
        {
            "text": "Hi Tom, how are you doing?",
            "expected": "ham"
        },
        {
            "text": "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot!",
            "expected": "spam"
        },
        {
            "text": "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?",
            "expected": "ham"
        },
        {
            "text": "URGENT! Your Mobile No. was awarded £200 Bonus Caller Prize on 1/08.",
            "expected": "spam"
        },
        {
            "text": "I HAVE A DATE ON SUNDAY WITH WILL!!",
            "expected": "ham"
        }
    ]
    
    print("Testing single predictions:")
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"text": test_case["text"]},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Test {i}:")
                print(f"  Text: {test_case['text'][:50]}...")
                print(f"  Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.4f}")
                print(f"  Expected: {test_case['expected']}")
                print()
            else:
                print(f"Test {i}: Failed with status {response.status_code}")
                print(f"  Response: {response.text}")
                print()
        except requests.exceptions.RequestException as e:
            print(f"Test {i}: Request failed - {e}")
            print()
    
    # Test batch prediction
    print("Testing batch predictions:")
    try:
        texts = [tc["text"] for tc in test_cases]
        response = requests.post(
            "http://localhost:8000/predict_batch",
            json=texts,
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"Batch prediction results:")
            for i, pred in enumerate(results["predictions"]):
                print(f"  {i+1}. {pred['prediction']} (confidence: {pred['confidence']:.4f})")
        else:
            print(f"Batch prediction failed with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Batch prediction request failed - {e}")

def show_api_docs():
    """Show information about API documentation"""
    print("=" * 50)
    print("STEP 4: API Documentation")
    print("=" * 50)
    
    print("API Documentation is available at:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("  - OpenAPI JSON: http://localhost:8000/openapi.json")
    print()
    print("API Endpoints:")
    print("  - GET  / - API information")
    print("  - GET  /health - Health check")
    print("  - POST /predict - Single prediction")
    print("  - POST /predict_batch - Batch predictions")
    print()

def cleanup(server_process):
    """Clean up resources"""
    if server_process:
        print("Stopping API server...")
        server_process.terminate()
        server_process.wait()

def main():
    """Main demo function"""
    print("Spam Classification API Demo")
    print("=" * 50)
    
    # Store original directory
    original_dir = os.getcwd()
    
    try:
        # Step 1: Train model
        if not train_model():
            print("Demo failed at training step")
            return
        
        # Step 2: Start API server
        server_process = start_api_server()
        if not server_process:
            print("Demo failed at server startup step")
            return
        
        # Step 3: Test API
        test_api()
        
        # Step 4: Show API docs
        show_api_docs()
        
        print("=" * 50)
        print("Demo completed successfully!")
        print("API server is running at http://localhost:8000")
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Keep server running
        try:
            server_process.wait()
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Cleanup
        cleanup(server_process)
        os.chdir(original_dir)

if __name__ == "__main__":
    main() 