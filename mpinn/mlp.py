import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import os
import time
from typing import Dict, List, Tuple, Optional
from mpi4py import MPI
from .mpiDataDistribution import MPIDD

class ActivationFunction:
    """Pluggable activation function class"""

    @staticmethod
    def relu(x):
        """ReLU activation function"""
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        """Tanh activation function"""
        return np.tanh(x)

    @staticmethod
    def relu_derivative(x):
        """ReLU derivative"""
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid derivative"""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh_derivative(x):
        """Tanh derivative"""
        return 1 - np.tanh(x) ** 2


class MLP:
    """1-hidden-layer Neural Network with MPI support for distributed training"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1,
                 activation: str = 'relu', learning_rate: float = 0.01,
                 random_seed: int = 42):
        """
        Initialize the neural network with MPI support

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units (default: 1 for regression)
            activation: Activation function name
            learning_rate: Learning rate for optimization
            random_seed: Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        # Set random seed (same for all processes for consistency)
        np.random.seed(random_seed)

        # Initialize weights and biases (same initialization for all processes)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        # Set activation function
        self.activation_name = activation
        self.activation_func, self.activation_derivative = self._get_activation_functions(activation)

        # Training history (only on rank 0)
        self.train_losses = []
        self.val_losses = []
        self.train_rmse = []
        self.val_rmse = []

    def _get_activation_functions(self, activation: str):
        """Get activation function and its derivative"""
        activations = {
            'relu': (ActivationFunction.relu, ActivationFunction.relu_derivative),
            'sigmoid': (ActivationFunction.sigmoid, ActivationFunction.sigmoid_derivative),
            'tanh': (ActivationFunction.tanh, ActivationFunction.tanh_derivative)
        }

        if activation not in activations:
            raise ValueError(f"Activation '{activation}' not supported. Choose from: {list(activations.keys())}")

        return activations[activation]

    def forward(self, X):
        """Forward propagation"""
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation_func(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for regression

        return self.a2

    def backward(self, X, y, output):
        """
        Backward propagation
            X:(n,d_in),
            W1:(d_in,h), b1:(1,h),
            W2:(h,d_out), b2:(1,d_out),
            output:(n,d_out)
        """
        n = X.shape[0]

        # Output layer gradients
        dz2 = 2.0 * (output - y)
        dW2 = (1 / n) * np.dot(self.a1.T, dz2)
        db2 = (1 / n) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.z1)
        dW1 = (1 / n) * np.dot(X.T, dz1)
        db1 = (1 / n) * np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def compute_loss(self, y_true, y_pred):
        """Compute mean squared error loss"""
        return np.mean((y_true - y_pred) ** 2)

    def compute_rmse(self, y_true, y_pred):
        """Compute Root Mean Square Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def ensure_2d_y(y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def compute_batch_grads(self, X: np.ndarray, y: np.ndarray):
        """Forward -> loss -> backward. Returns grads and loss."""
        y = self.ensure_2d_y(y)
        y_pred = self.forward(X)
        loss = self.compute_loss(y, y_pred)
        dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "loss": loss}

    def update_model_parameters(self, W1, b1, W2, b2):
        """Update the model's parameters values"""
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        return str(f"Update: W1 -> {W1}, b1 -> {b1}, W2 -> {W2}, b2 -> {b2}")


