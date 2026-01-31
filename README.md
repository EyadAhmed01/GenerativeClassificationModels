# Generative Classification Models

A comprehensive machine learning project implementing four classification models from scratch: Gaussian Generative Classifier, Naive Bayes, Decision Tree, and Random Forest.

## Overview

This project demonstrates the implementation of fundamental machine learning classification algorithms from scratch using NumPy and basic Python libraries. Each model is implemented with proper hyperparameter tuning, evaluation metrics, and detailed analysis.

## Models Implemented

### 1. Gaussian Generative Classifier with Shared Covariance
- **Dataset**: sklearn digits dataset (handwritten digits 0-9)
- **Features**: 64 features (8x8 pixel images)
- **Implementation**: Gaussian Discriminant Analysis (GDA) with shared covariance matrix
- **Key Features**:
  - Maximum likelihood estimation for class priors and means
  - Shared covariance matrix with regularization
  - Hyperparameter tuning for regularization parameter λ
  - Comprehensive evaluation with confusion matrix analysis

### 2. Naive Bayes Classifier
- **Dataset**: Adult dataset (categorical features)
- **Features**: 8 categorical features (workclass, education, marital.status, occupation, relationship, race, sex, native.country)
- **Implementation**: Categorical Naive Bayes with Laplace smoothing
- **Key Features**:
  - Laplace smoothing for handling unseen feature values
  - Hyperparameter tuning for smoothing parameter α
  - Feature selection analysis
  - Comparison with sklearn's MultinomialNB and CategoricalNB
  - Probability distribution analysis

### 3. Decision Tree Classifier
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) dataset
- **Features**: 30 continuous features
- **Implementation**: Binary decision tree with information gain
- **Key Features**:
  - Information gain as split criterion
  - Entropy as impurity measure
  - Hyperparameter tuning for max_depth and min_samples_split
  - Feature importance analysis
  - Overfitting analysis
  - Tree complexity visualization

### 4. Random Forest
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) dataset
- **Features**: 30 continuous features
- **Implementation**: Ensemble of decision trees with bootstrap sampling
- **Key Features**:
  - Bootstrap sampling (sampling with replacement)
  - Random feature subset selection at each split
  - Majority voting for predictions
  - Hyperparameter tuning for number of trees and max_features
  - Comparison with single decision tree performance

## Requirements

### Python Libraries
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Data visualization
- `scikit-learn` - Dataset loading and utilities

### Datasets
- sklearn digits dataset (included with sklearn)
- Adult dataset (`adult.csv` file required)
- sklearn breast cancer dataset (included with sklearn)

## Project Structure

```
GenerativeClassificationModels/
├── main.ipynb          # Main notebook with all implementations
├── main.pdf            # PDF version of the notebook
├── adult.csv           # Adult dataset (required for Naive Bayes)
└── README.md           # This file
```

## Usage

1. **Install Dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

2. **Download Dataset**
   - The Adult dataset (`adult.csv`) should be placed in the project root directory
   - sklearn datasets (digits, breast_cancer) are automatically downloaded

3. **Run the Notebook**
   - Open `main.ipynb` in Jupyter Notebook or JupyterLab
   - Execute cells sequentially to run each model implementation

## Key Features

### Data Preprocessing
- Standardization (zero mean, unit variance) for continuous features
- Stratified train/validation/test splits (70/15/15)
- Missing value handling for categorical data
- Feature encoding and transformation

### Model Evaluation
- Accuracy, precision, recall, F1-score
- Confusion matrix analysis
- Per-class metrics
- Training vs validation performance analysis
- Probability calibration analysis

### Hyperparameter Tuning
- Grid search on validation sets
- Regularization parameter tuning (Gaussian classifier)
- Smoothing parameter tuning (Naive Bayes)
- Tree depth and split criteria tuning (Decision Tree)
- Ensemble size and feature subset tuning (Random Forest)

### Analysis & Visualization
- Confusion matrices
- Feature importance plots
- Overfitting analysis
- Probability distribution histograms
- Performance comparison charts

## Results Summary

### Gaussian Generative Classifier
- Implements GDA with shared covariance matrix
- Regularization prevents overfitting
- Achieves good performance on digit classification task

### Naive Bayes
- Handles categorical data effectively
- Feature selection analysis reveals most important features
- Comparable performance to sklearn implementations

### Decision Tree
- Achieves high accuracy on breast cancer classification
- Feature importance analysis identifies key diagnostic features
- Overfitting analysis shows optimal depth selection

### Random Forest
- Outperforms single decision tree through ensemble learning
- Reduces variance through averaging multiple trees
- Demonstrates bias-variance tradeoff benefits

## Implementation Details

All models are implemented from scratch using only NumPy and basic Python, demonstrating:
- Understanding of mathematical foundations
- Proper implementation of algorithms
- Best practices in machine learning (train/val/test splits, hyperparameter tuning)
- Comprehensive evaluation and analysis

## Notes

- All implementations use random seeds for reproducibility
- Models follow sklearn-like API (fit/predict methods)
- Comprehensive comments and documentation throughout
- Detailed analysis and visualizations for each model

## License

This project is for educational purposes, demonstrating machine learning algorithm implementations from scratch.
