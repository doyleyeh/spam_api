# Spam Classification API

A machine learning API for classifying text messages as spam or ham using DistilBERT.

## Features

- **DistilBERT-based Classification**: Uses the lightweight DistilBERT model for efficient text classification
- **FastAPI Backend**: High-performance REST API with automatic documentation
- **Docker Support**: Containerized deployment for easy scaling
- **Model Training Pipeline**: Complete training and evaluation scripts
- **Comprehensive Testing**: Unit tests for API endpoints and model functionality

## Project Structure

```
spam_api/
├── README.md                 # Project overview, setup, usage instructions
├── api/
│   ├── main.py               # FastAPI API endpoints
│   ├── requirements.txt      # API dependencies
│   └── Dockerfile            # Docker instructions for deployment
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
git clone <repo-url>
cd spam_api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r api/requirements.txt
```

### 2. Train the Model

```bash
# Run the training script
python scripts/train_model.py
```

### 3. Start the API

```bash
# Start the FastAPI server
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Use the API

The API will be available at `http://localhost:8000`

- **API Documentation**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

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

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Docker Deployment

```bash
# Build the Docker image
docker build -t spam-classifier-api ./api

# Run the container
docker run -p 8000:8000 spam-classifier-api
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
- **Epochs**: 5

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

<!-- ## Performance

The model achieves:

- **Accuracy**: ~95%
- **Precision**: ~94%
- **Recall**: ~96%
- **F1-Score**: ~95% -->

<!-- ## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request -->

<!-- ## License

MIT License - see LICENSE file for details. -->
