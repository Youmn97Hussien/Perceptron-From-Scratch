from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def Create_Dataset():
    Featuers, Labels = make_classification(n_samples=10000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
    Labels = np.where(Labels == 0, -1, 1)  # Convert labels to -1 and 1
    return Featuers, Labels

def Standardize_Data(Featuers):
    mean = np.mean(Featuers, axis=0)
    std_dev = np.std(Featuers, axis=0)
    Featuers = (Featuers - mean) / std_dev
    return Featuers

def splitt_preprocess(Featuers, Labels):
    Featuers = Standardize_Data(Featuers)
    Train_Featuers, Test_Featuers, Train_Labels, Test_Labels = train_test_split(Featuers, Labels, test_size=0.3, random_state=42)
    return Train_Featuers, Test_Featuers, Train_Labels, Test_Labels

def TrainPerceptron(Train_Featuers, Train_Labels):
    LearningRate = 0.001
    Weights = np.zeros(Train_Featuers.shape[1])  # Initialize weights to zeros
    bias = 0
    max_epochs = 150
    MSEPlot = []
    current_epoch = 0

    while current_epoch < max_epochs:
        current_epoch += 1
        num_errors = 0
        for x, y in zip(Train_Featuers, Train_Labels):
            # Update rule if there is a misclassification
            if y * (np.dot(x, Weights) + bias) <= 0:
                Weights += LearningRate * y * x
                bias += LearningRate * y
                num_errors += 1
        MSEPlot.append(num_errors)
        # Stop early if no errors
        if num_errors == 0:
            break

    return Weights, bias, MSEPlot

def predict_Perceptron(Weights, bias, Test_Featuers):
    PredictZ = np.dot(Test_Featuers, Weights) + bias
    predictedtestLabel = np.where(PredictZ >= 0.0, 1, -1)
    accuracy = (np.sum(Test_Labels == predictedtestLabel) / len(Test_Labels)) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return predictedtestLabel

def Visualization(predictedtestLabel, Test_Labels, MSEPlot):
    plt.figure(figsize=(10, 6))
    plt.plot(Test_Labels, label='True Labels', color='blue', alpha=0.7)
    plt.plot(predictedtestLabel, label='Predicted Labels', color='green', alpha=0.7)
    plt.title('True vs Predicted Labels')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(MSEPlot, label='Errors vs Epochs', color='red', linewidth=2)  # Continuous curve
    plt.title('Performance (Errors vs Epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Misclassifications')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show(block=True)

Featuers, Labels = Create_Dataset()
Train_Featuers, Test_Featuers, Train_Labels, Test_Labels = splitt_preprocess(Featuers, Labels)
Weights, bias, MSEPlot = TrainPerceptron(Train_Featuers, Train_Labels)
predictedtestLabel = predict_Perceptron(Weights, bias, Test_Featuers)
Visualization(predictedtestLabel, Test_Labels, MSEPlot)
