from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import joblib
import os
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Spam Classification API",
    description="A machine learning API for classifying text messages as spam or ham using DistilBERT",
    version="1.0.0"
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    text: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Global variables for model and tokenizer
model = None
tokenizer = None
label_encoder = None

def load_model():
    """Load the trained model, tokenizer, and label encoder using Hugging Face's from_pretrained method"""
    global model, tokenizer, label_encoder
    
    try:
        # Load the trained model using Hugging Face's methods
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "spam_classifier")
        if os.path.exists(model_path):
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            
            # Load label encoder separately
            label_encoder_path = os.path.join(model_path, 'label_encoder.pkl')
            label_encoder = joblib.load(label_encoder_path)
            
            logger.info("Model loaded successfully using Hugging Face's from_pretrained method")
        else:
            logger.warning("Model directory not found. Please train the model first.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Spam Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_spam(request: PredictionRequest):
    """Predict whether a text message is spam or legitimate"""
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model has been trained and is available."
        )
    
    try:
        # Preprocess the input text
        inputs = tokenizer(
            request.text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Convert prediction to label
        if label_encoder is not None:
            prediction = label_encoder.inverse_transform([predicted_class])[0]
        else:
            prediction = "spam" if predicted_class == 1 else "ham"
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            text=request.text
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

@app.post("/predict_batch")
async def predict_batch(texts: list[str]):
    """Predict multiple text messages at once"""
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model has been trained and is available."
        )
    
    try:
        results = []
        
        for text in texts:
            # Preprocess the input text
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Convert prediction to label
            if label_encoder is not None:
                prediction = label_encoder.inverse_transform([predicted_class])[0]
            else:
                prediction = "spam" if predicted_class == 1 else "ham"
            
            results.append({
                "text": text,
                "prediction": prediction,
                "confidence": confidence
            })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during batch prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 