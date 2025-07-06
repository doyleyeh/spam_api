# Spam Classification API

A machine learning API for classifying text messages as spam or ham using DistilBERT. Supports both traditional server deployment and AWS Lambda serverless deployment.

## Features

- **DistilBERT-based Classification**: Uses the lightweight DistilBERT model for efficient text classification
- **FastAPI Backend**: High-performance REST API with automatic documentation
- **AWS Lambda Support**: Serverless deployment with container images
- **Docker Support**: Containerized deployment for easy scaling
- **Model Training Pipeline**: Complete training and evaluation scripts
- **Comprehensive Testing**: Unit tests for API endpoints and model functionality

## Project Structure

```
spam_api/
├── README.md                 # Project overview, setup, usage instructions
├── api/
│   ├── main.py               # FastAPI API endpoints
│   ├── lambda_handler.py     # AWS Lambda handler
│   ├── requirements.txt      # API dependencies
│   ├── Dockerfile            # Docker instructions for server deployment
│   └── Dockerfile.lambda     # Docker instructions for AWS Lambda
├── models/
│   └── spam_classifier/      # Trained model directory (Hugging Face format)
├── notebooks/
│   └── train_model.ipynb     # Analysis & training notebooks
├── scripts/
│   └── train_model.py        # Training logic script
└── tests/
    └── test_api.py           # Unit tests
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/doyleyeh/spam_api.git
cd spam_api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model through script or notebooks

```bash
# Run the training script
python scripts/train_model.py
```

### 3. Start the API (Local Development)

```bash
# Start the FastAPI server
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

- **API Documentation**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

## Deployment Options

### Option 1: Traditional Server Deployment

#### Docker Deployment

```bash
# Build the Docker image (from project root)
docker build -f api/Dockerfile -t spam-classifier-api .

# Run the container
docker run -d -p 8000:8000 --name spam-api-container spam-classifier-api
```

### Option 2: AWS Lambda Deployment

#### Prerequisites

1. **AWS CLI** installed and configured
2. **Docker** installed
3. **AWS ECR** repository created

#### Build and Deploy to AWS Lambda

Steps:

1. Create ECR repository (if not exists)
2. Get ECR login token
3. Build Lambda container image
4. Tag the image
5. Push to ECR
6. Create Lambda function (via AWS Console or CLI)

## API Endpoints

### POST /predict

Classify a text message as spam or ham.

**Request Body:**

```json
{
  "text": "Your message here"
}
```

**Response:**

```json
{
  "prediction": "spam",
  "confidence": 0.95,
  "text": "Your message here"
}
```

### POST /predict_batch

Classify multiple text messages at once.

**Request Body:**

```json
["First message", "Second message", "Third message"]
```

**Response:**

```json
{
  "predictions": [
    {
      "text": "First message",
      "prediction": "ham",
      "confidence": 0.98
    },
    {
      "text": "Second message",
      "prediction": "spam",
      "confidence": 0.95
    },
    {
      "text": "Third message",
      "prediction": "ham",
      "confidence": 0.87
    }
  ]
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Model Training

The model training process includes:

1. **Data Preprocessing**: Text cleaning and tokenization
2. **Model Training**: Fine-tuning DistilBERT on spam classification
3. **Evaluation**: Performance metrics and validation
4. **Serialization**: Saving the trained model

### Training Configuration

- **Model**: `distilbert-base-uncased`
- **Max Sequence Length**: 512 tokens
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 3

## Testing

### Local Testing

### Docker Testing

```bash
# Build and test with Docker
docker build -f api/Dockerfile -t spam-classifier-api .
docker run -p 8000:8000 spam-classifier-api

# Test endpoints
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "URGENT! You have won a prize!"}'
```

### Lambda Testing

```bash
# Build and test Lambda function locally
docker build -t spam-classifier-lambda -f .\api\Dockerfile.lambda .
docker run -p 9000:8080 spam-classifier-lambda

# Test with Lambda Runtime Interface Emulator
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{
    "version": "2.0",
    "routeKey": "GET /health",
    "rawPath": "/health",
    "rawQueryString": "",
    "headers": {},
    "requestContext": {
        "http": {
            "method": "GET",
            "path": "/health",
            "sourceIp": "127.0.0.1"
        }
    },
    "isBase64Encoded": false
}'
```

## License

MIT License - see LICENSE file for details.
