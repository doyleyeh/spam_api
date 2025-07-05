#!/usr/bin/env python3
"""
Unit tests for the Spam Classification API

This module contains comprehensive tests for:
- API endpoints
- Model loading and prediction
- Error handling
- Data validation
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient


# Add the api directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from main import app

# Create test client
client = TestClient(app)

class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Spam Classification API"
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
        # model_loaded will be False if no model is available
        assert isinstance(data["model_loaded"], bool)
    
    def test_predict_endpoint_without_model(self):
        """Test prediction endpoint when model is not loaded"""
        # This test assumes no model is loaded
        response = client.post("/predict", json={"text": "Hello world"})
        # Should return 503 if model is not loaded
        if response.status_code == 503:
            data = response.json()
            assert "detail" in data
            assert "Model not loaded" in data["detail"]
        else:
            # If model is loaded, should return prediction
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "text" in data
            assert data["text"] == "Hello world"
            assert data["prediction"] in ["spam", "ham"]
            assert 0 <= data["confidence"] <= 1
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction endpoint with invalid input"""
        # Test with missing text field
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error
        
        # Test with empty text
        response = client.post("/predict", json={"text": ""})
        if response.status_code == 200:
            # If model is loaded, should handle empty text
            data = response.json()
            assert "prediction" in data
        else:
            # Should return validation error or 503
            assert response.status_code in [422, 503]
    
    def test_predict_batch_endpoint(self):
        """Test batch prediction endpoint"""
        texts = ["Hello world", "URGENT! You have won a prize!"]
        response = client.post("/predict_batch", json=texts)
        
        if response.status_code == 503:
            # Model not loaded
            data = response.json()
            assert "detail" in data
            assert "Model not loaded" in data["detail"]
        else:
            # Model loaded
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
            
            for pred in data["predictions"]:
                assert "text" in pred
                assert "prediction" in pred
                assert "confidence" in pred
                assert pred["prediction"] in ["spam", "ham"]
                assert 0 <= pred["confidence"] <= 1
    
    def test_predict_batch_empty_list(self):
        """Test batch prediction with empty list"""
        response = client.post("/predict_batch", json=[])
        if response.status_code == 503:
            # Model not loaded
            data = response.json()
            assert "detail" in data
        else:
            # Model loaded
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 0

class TestModelFunctionality:
    """Test cases for model functionality"""
    
    def test_model_loading(self):
        """Test model loading functionality"""
        from main import load_model
        
        # Test loading when model file doesn't exist
        result = load_model()
        # Should return False if model file doesn't exist
        assert isinstance(result, bool)
    
    def test_prediction_logic(self):
        """Test prediction logic with mock data"""
        # This test would require a trained model to be present
        # For now, we'll test the structure of the prediction response
        test_text = "This is a test message"
        
        response = client.post("/predict", json={"text": test_text})
        
        if response.status_code == 200:
            data = response.json()
            # Check response structure
            required_fields = ["prediction", "confidence", "text"]
            for field in required_fields:
                assert field in data
            
            # Check data types
            assert isinstance(data["prediction"], str)
            assert isinstance(data["confidence"], (int, float))
            assert isinstance(data["text"], str)
            
            # Check value ranges
            assert data["prediction"] in ["spam", "ham"]
            assert 0 <= data["confidence"] <= 1
            assert data["text"] == test_text

class TestDataValidation:
    """Test cases for data validation"""
    
    def test_valid_text_input(self):
        """Test with valid text input"""
        valid_texts = [
            "Hello world",
            "This is a normal message",
            "URGENT! You have won a prize!",
            "A" * 1000,  # Long text
            "Short",     # Short text
            "Text with numbers 123 and symbols !@#",
            "Unicode text: 你好世界",
            "Text with newlines\nand tabs\t",
        ]
        
        for text in valid_texts:
            response = client.post("/predict", json={"text": text})
            if response.status_code == 200:
                data = response.json()
                assert data["text"] == text
            elif response.status_code == 503:
                # Model not loaded, which is acceptable
                pass
            else:
                # Should not get validation errors for valid text
                assert response.status_code not in [422, 400]
    
    def test_invalid_input_types(self):
        """Test with invalid input types"""
        invalid_inputs = [
            {"text": None},
            {"text": 123},
            {"text": []},
            {"text": {}},
            {"text": True},
        ]
        
        for invalid_input in invalid_inputs:
            response = client.post("/predict", json=invalid_input)
            # Should return validation error
            assert response.status_code == 422

class TestErrorHandling:
    """Test cases for error handling"""
    
    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        response = client.post("/predict", data="invalid json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422
    
    def test_wrong_content_type(self):
        """Test handling of wrong content type"""
        response = client.post("/predict", data="Hello world", headers={"Content-Type": "text/plain"})
        assert response.status_code == 422
    
    def test_missing_content_type(self):
        """Test handling of missing content type"""
        response = client.post("/predict", data='{"text": "Hello world"}')
        assert response.status_code == 422

class TestAPIDocumentation:
    """Test cases for API documentation"""
    
    def test_openapi_schema(self):
        """Test that OpenAPI schema is available"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_docs_endpoint(self):
        """Test that docs endpoint is available"""
        response = client.get("/docs")
        assert response.status_code == 200
        # Should return HTML content
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_endpoint(self):
        """Test that redoc endpoint is available"""
        response = client.get("/redoc")
        assert response.status_code == 200
        # Should return HTML content
        assert "text/html" in response.headers.get("content-type", "")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 