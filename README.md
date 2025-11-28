# Machine Learning for Lunar Crater Classification

This repository contains Python Jupyter notebooks and data files for classifying craters on the Moon from grayscale images using different machine learning models, including convolutional and fully connected neural networks, as well as random forests.

## Overview

The project implements:

* **Convolutional Neural Network (CNN)** — LeNet-inspired architecture for crater classification images (`Clasification_problem_CNN.ipynb`).
* **Multilayer Perceptron (MLP)** — fully connected network for crater classification images (`Clasification_problem_MLP.ipynb`).
* **Random Forest Classifier** — baseline model with hyperparameter tuning for crater classification images(`Clasification_problem_forest.ipynb`).
* **Data Preprocessing Utilities** — binarization, normalization, and handling class imbalance (oversampling/undersampling).
* **Evaluation and Visualization** — training curves, F1 scores, confusion matrices, and K-Fold cross-validation results.

## Files

* `Clasification_problem_CNN.ipynb` — CNN model implementation and training notebook
* `Clasification_problem_MLP.ipynb` — MLP model implementation and training notebook
* `Clasification_problem_forest.ipynb` — Random Forest model implementation and training notebook
* `Xtrain1.npy` — training dataset images
* `Xtrain1_extra.npy` — additional dataset images for semi-supervised training
* `Ytrain1.npy` — training dataset labels
* `Xtest1.npy` — test dataset images
  
## Requirements

### Core Software

* Python 3.12 (recommended)

### Mandatory Python Packages

* numpy
* matplotlib
* scikit-learn
* torch
* imbalanced-learn
* keras

## Outputs 
- `y_test.npy`: predictions of Xtest1.npy from the best model (CNN)  
