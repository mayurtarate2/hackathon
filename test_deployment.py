#!/usr/bin/env python3
"""
Deployment verification script for HackRX 6.0 API
Tests the deployed API endpoint to ensure it's working correctly
"""

import requests
import json
import sys
import time
from typing import Dict, Any

def test_deployed_api(base_url: str, auth_token: str) -> bool:
    """Test the deployed API endpoint"""
    
    print(f"🔍 Testing deployed API at: {base_url}")
    print("=" * 60)
    
    # Test 1: Health Check
    print("1️⃣ Testing health endpoint...")
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Health check: PASS")
            print(f"   Response: {health_response.json()}")
        else:
            print(f"❌ Health check: FAIL (Status: {health_response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Health check: FAIL (Error: {str(e)})")
        return False
    
    # Test 2: Root endpoint  
    print("\n2️⃣ Testing root endpoint...")
    try:
        root_response = requests.get(base_url, timeout=10)
        if root_response.status_code == 200:
            print("✅ Root endpoint: PASS")
        else:
            print(f"❌ Root endpoint: FAIL (Status: {root_response.status_code})")
    except Exception as e:
        print(f"❌ Root endpoint: FAIL (Error: {str(e)})")
    
    # Test 3: Authentication Test
    print("\n3️⃣ Testing authentication...")
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    test_payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sp=r&st=2025-01-25T12:29:55Z&se=2025-01-26T20:29:55Z&spr=https&sv=2022-11-02&sr=b&sig=B%2F3nSCITJHpnQjafZeYXKW%2BOflzNKqMd8e6kDQNiLgI%3D",
        "questions": ["What is the grace period for premium payment?"]
    }
    
    try:
        print("   Sending test request...")
        start_time = time.time()
        
        api_response = requests.post(
            f"{base_url}/hackrx/run",
            headers=headers,
            json=test_payload,
            timeout=60  # Allow up to 60 seconds for processing
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"   Processing time: {processing_time:.2f} seconds")
        
        if api_response.status_code == 200:
            print("✅ API request: PASS")
            response_data = api_response.json()
            
            # Validate response structure
            if "answers" in response_data and isinstance(response_data["answers"], list):
                print("✅ Response format: CORRECT")
                print(f"   Answer received: {response_data['answers'][0][:100]}...")
                
                if processing_time < 30:
                    print("✅ Response time: ACCEPTABLE (<30s)")
                else:
                    print("⚠️ Response time: SLOW (>30s)")
                
                return True
            else:
                print("❌ Response format: INCORRECT")
                print(f"   Response: {response_data}")
                return False
                
        else:
            print(f"❌ API request: FAIL (Status: {api_response.status_code})")
            print(f"   Response: {api_response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ API request: TIMEOUT (>60 seconds)")
        return False
    except Exception as e:
        print(f"❌ API request: FAIL (Error: {str(e)})")
        return False

def main():
    """Main deployment test function"""
    
    if len(sys.argv) != 2:
        print("Usage: python test_deployment.py <DEPLOYED_API_URL>")
        print("\nExample:")
        print("python test_deployment.py https://your-app.railway.app")
        print("python test_deployment.py https://your-app.onrender.com")
        print("python test_deployment.py https://your-app-xxx.run.app")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    auth_token = "880b4911f53f0dc33bb443bfc2c5831f87db7bc9d8bf084d6f42acb6918b02f7"
    
    print("🚀 HackRX 6.0 API Deployment Test")
    print("=" * 60)
    print(f"Testing URL: {base_url}")
    print(f"Auth Token: {auth_token[:20]}...")
    print("=" * 60)
    
    success = test_deployed_api(base_url, auth_token)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 DEPLOYMENT TEST: SUCCESS!")
        print("✅ Your API is ready for HackRX 6.0 submission!")
        print(f"\nSubmission Details:")
        print(f"- API URL: {base_url}")
        print(f"- Endpoint: POST {base_url}/hackrx/run")
        print(f"- Auth Token: {auth_token}")
        print(f"- Documentation: {base_url}/docs")
    else:
        print("❌ DEPLOYMENT TEST: FAILED!")
        print("Please check the deployment logs and fix the issues.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
