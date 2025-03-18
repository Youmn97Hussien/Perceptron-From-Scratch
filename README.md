# Perceptron Implementation from Scratch 🧠

This repository contains an implementation of the Perceptron algorithm using Python. The model achieves **75.30% accuracy** on the given dataset. Below are the details about the code, its functionalities, and its usage.

## Table of Contents 📚
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Code Overview](#code-overview)
- [Usage](#usage)
- [Visualization](#visualization)
- [Results](#results)

## Introduction 🧐

The Perceptron is one of the simplest and most effective algorithms for binary classification. This implementation demonstrates how the Perceptron algorithm works with a dataset generated synthetically and goes through:
1. Dataset generation
2. Preprocessing
3. Model training
4. Prediction
5. Visualization

## Requirements 🛠️

To run this code, ensure you have the following libraries installed:
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install them using pip:
```bash
pip install numpy matplotlib scikit-learn

## Code Overview 🖥️

### Functions:
- **`Create_Dataset`**:  
  Generates a synthetic dataset with 10,000 samples and 10 features.

- **`Standardize_Data`**:  
  Standardizes the dataset using mean and standard deviation.

- **`splitt_preprocess`**:  
  Splits the standardized dataset into training and testing sets.

- **`TrainPerceptron`**:  
  Trains the Perceptron model using gradient descent and returns:
  - Best weights
  - Bias
  - MSE (Mean Squared Error) plot

- **`Cost_Function`**:  
  Calculates the Mean Squared Error (MSE) of the model.

- **`predict_Perceptron`**:  
  Predicts labels for test data using the trained weights and bias.  
  Also calculates the accuracy of the model.

- **`Visualization`**🎨:  
  Provides visualizations, including:
  1. **True vs. Predicted Labels** 📊  
     A plot comparing the true labels and predicted labels for the test dataset.
  2. **MSE Curve**  📈  
     A curve showcasing how the Mean Squared Error decreases over epochs during training.


## Results 🏆

The Perceptron model achieves an **accuracy of 75.93%** on the synthetic test dataset.

This demonstrates the effectiveness of Perceptron as a simple and interpretable model for binary classification. 🎉
