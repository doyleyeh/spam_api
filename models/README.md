# Models Directory

This directory stores trained machine learning models for the spam classification API.

## Contents

- `spam_classifier/` - Directory containing the trained DistilBERT model for spam classification (created after training)
  - `config.json` - Model configuration
  - `pytorch_model.bin` - Model weights
  - `tokenizer.json` - Tokenizer configuration
  - `vocab.txt` - Vocabulary file
  - `label_encoder.pkl` - Label encoder for class mapping

## Model Information

The trained model includes:

- **Model**: DistilBERT fine-tuned for binary classification
- **Tokenizer**: DistilBERT tokenizer for text preprocessing
- **Label Encoder**: Scikit-learn LabelEncoder for class mapping
- **Classes**: "spam" and "ham"

## Usage

The model is automatically loaded by the API when the server starts. The model file is created after running the training script:

```bash
python scripts/train_model.py
```

## File Format

The model is saved using Hugging Face's standard format:

- Model weights and configuration in standard Hugging Face format
- Tokenizer files in standard Hugging Face format
- Label encoder saved separately as a pickle file

## Note

This directory is included in `.gitignore` to avoid committing large model files to version control. The model will be created locally when you run the training script.
