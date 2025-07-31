#!/usr/bin/env python3

import requests
import json

# API Configuration
BASE_URL = "http://localhost:8000"
API_TOKEN = "880b4911f53f0dc33bb443bfc2c5831f87db7bc9d8bf084d6f42acb6918b02f7"

def test_api():
    """Test the /hackrx/run endpoint with sample data"""
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }
    
    # Test payload (using the sample from the problem statement)
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    print("🚀 Testing HackRX API endpoint...")
    print(f"📡 Sending request to: {BASE_URL}/hackrx/run")
    print(f"📝 Questions: {len(payload['questions'])}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            json=payload,
            timeout=300  # 5 minutes timeout for document processing
        )
        
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Received answers:")
            print("=" * 80)
            
            for i, (question, answer) in enumerate(zip(payload['questions'], result['answers'])):
                print(f"\n🤔 Question {i+1}: {question}")
                print(f"💡 Answer: {answer}")
                print("-" * 40)
                
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def test_health():
    """Test health endpoints"""
    print("\n🏥 Testing health endpoints...")
    
    try:
        # Test root endpoint
        response = requests.get(f"{BASE_URL}/")
        print(f"Root endpoint: {response.status_code} - {response.json() if response.status_code == 200 else response.text}")
        
        # Test health endpoint
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health endpoint: {response.status_code} - {response.json() if response.status_code == 200 else response.text}")
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")

if __name__ == "__main__":
    print("🧪 HackRX API Test Suite")
    print("=" * 50)
    
    # Test health first
    test_health()
    
    # Test main API
    test_api()
    
    print("\n🏁 Test completed!")
