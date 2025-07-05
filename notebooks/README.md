# Notebooks Directory

This directory contains Jupyter notebooks for interactive model training and analysis.

## Available Notebooks

### train_model.ipynb

A comprehensive notebook for training the spam classification model with DistilBERT.

**Features:**

- Data loading and exploration
- Data preprocessing and visualization
- Model setup and training
- Performance evaluation
- Model testing and saving

## Usage

1. Install Jupyter dependencies:

   ```bash
   pip install jupyter matplotlib seaborn
   ```

2. Start Jupyter Lab or Jupyter Notebook:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

3. Navigate to this directory and open the desired notebook

## Alternative Training Method

If you prefer to run training without Jupyter, use the script:

```bash
python scripts/train_model.py
```

## Requirements

The notebooks require the same dependencies as the main project. See the root `requirements.txt` file for details.
