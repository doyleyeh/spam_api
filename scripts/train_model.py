#!/usr/bin/env python3
"""
Spam Classification Model Training Script

This script trains a DistilBERT model for spam classification using a dataset
of spam ham.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpamDataset(Dataset):
    """Custom Dataset for spam classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_or_create_dataset():
    """Load the real spam-ham dataset from CSV."""
    logger.info("Loading spam-ham dataset from data/spam_ham_dataset.csv ...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'spam_ham_dataset.csv'))
    # Use only the columns we need
    df = df[['text', 'label']]
    logger.info(f"Loaded dataset with {len(df)} samples. Class distribution:\n{df['label'].value_counts()}")
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[List[str], List[int], LabelEncoder]:
    """Preprocess the data and encode labels"""
    logger.info("Preprocessing data...")
    
    # Clean text (basic cleaning)
    df['text'] = df['text'].astype(str).str.strip()
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df['label'])
    
    logger.info(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return df['text'].tolist(), labels_encoded.tolist(), label_encoder

def train_model(model, train_loader, val_loader, device, num_epochs=5):
    """Train the model"""
    logger.info("Starting model training...")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        logger.info(f'Average training loss: {avg_train_loss:.4f}')
        
        # Validation phase
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_eval_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                total_eval_accuracy += (preds == labels).sum().item()
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_val_accuracy = total_eval_accuracy / len(val_loader.dataset)
        avg_val_loss = total_eval_loss / len(val_loader)
        
        logger.info(f'Validation Loss: {avg_val_loss:.4f}')
        logger.info(f'Validation Accuracy: {avg_val_accuracy:.4f}')
        
        if avg_val_accuracy > best_accuracy:
            best_accuracy = avg_val_accuracy
            logger.info(f'New best accuracy: {best_accuracy:.4f}')
    
    return predictions, true_labels

def evaluate_model(predictions, true_labels, label_encoder):
    """Evaluate the model performance"""
    logger.info("Evaluating model...")
    
    # Convert numeric predictions back to labels
    pred_labels = label_encoder.inverse_transform(predictions)
    true_labels_decoded = label_encoder.inverse_transform(true_labels)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels_decoded, pred_labels))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels_decoded, pred_labels)
    print(cm)
    
    # Print accuracy
    accuracy = accuracy_score(true_labels_decoded, pred_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

def save_model(model, tokenizer, label_encoder, model_path):
    """Save the trained model using Hugging Face's save_pretrained method"""
    logger.info(f"Saving model to {model_path}")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save model and tokenizer using Hugging Face's methods
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Save label encoder separately
    label_encoder_path = os.path.join(model_path, 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)
    
    logger.info("Model saved successfully using Hugging Face's save_pretrained method!")

def main():
    """Main training function"""
    logger.info("Starting spam classification model training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load or create dataset
    df = load_or_create_dataset()
    
    # Preprocess data
    texts, labels, label_encoder = preprocess_data(df)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Initialize tokenizer and model
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )
    model.to(device)
    
    # Create datasets
    train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
    val_dataset = SpamDataset(val_texts, val_labels, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Train model
    predictions, true_labels = train_model(model, train_loader, val_loader, device)
    
    # Evaluate model
    evaluate_model(predictions, true_labels, label_encoder)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "spam_classifier")
    save_model(model, tokenizer, label_encoder, model_path)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 
