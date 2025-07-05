# Data Directory

This directory contains the dataset files for training the spam classification model.

## Contents

- `spam_ham_dataset.csv` - The main dataset file containing spam and ham messages

## Dataset Format

The dataset should be a CSV file with the following columns:

- `text` - The message content
- `label` - The class label ("spam" or "ham")

## Usage

The training script automatically loads the dataset from this directory. Make sure to place the dataset file here before running the training.

## Note

This directory is currently included in `.gitignore` but commented.
