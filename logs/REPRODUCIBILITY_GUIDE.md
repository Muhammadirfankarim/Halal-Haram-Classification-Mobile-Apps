
# Halal/Haram Food Detection System - Reproducibility Guide

## Overview
This guide provides instructions to reproduce the halal/haram food detection experiment.

## Requirements
- Python 3.8+
- TensorFlow 2.15.0
- Transformers library (for BERT models)
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn

## Dataset
- File: cleaned_dataset.csv
- Expected columns: text, label
- Size: ~40,000 samples

## Configuration
The experiment uses the following key parameters:
- Random seed: 42
- Test size: 0.2
- Validation size: 0.2
- Max vocabulary: 10000
- Sequence length: 128
- Batch size: 64
- Learning rate: 5e-05

## Models
1. **CNN1D**: Baseline convolutional neural network
   - Embedding dimension: 128
   - Dropout: 0.5

2. **BERT/MobileBERT**: Transformer-based model
   - Base model: distilbert-base-uncased
   - Max length: 96
   - Dropout: 0.3

## Steps to Reproduce
1. Install required dependencies
2. Prepare the dataset in the specified format
3. Run the notebook cells in order
4. Results will be saved in the 'results/' directory
5. Models will be saved in the 'models/' directory

## Expected Results
The experiment should produce:
- Model comparison metrics
- TensorFlow Lite models for mobile deployment
- Statistical significance testing results
- Comprehensive visualizations

## Output Files
- Model files: models/
- Results: results/
- Logs: logs/
- Visualizations: results/*.png

## Contact
For questions about reproduction, please refer to the experiment log file.

Generated on: 2025-10-25 13:12:10
