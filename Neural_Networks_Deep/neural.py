# neural.py - Deep Neural Network Implementation from Scratch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class DeepNeuralNetwork:
    """
    A deep neural network implementation from scratch using only NumPy.
    Supports multiple hidden layers with customizable sizes.
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, random_seed=42):
        """
        Initialize the deep neural network.
        
        Args:
            layer_sizes (list): List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            learning_rate (float): Learning rate for gradient descent
            random_seed (int): Seed for reproducible results
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        np.random.seed(random_seed)
        
        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        
        for i in range(1, self.num_layers):
            # Xavier/Glorot initialization for better convergence
            self.weights[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i-1])
            self.biases[f'b{i}'] = np.zeros((1, layer_sizes[i]))
        
        # Store training history
        self.cost_history = []
        self.accuracy_history = []
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU function"""
        return (z > 0).astype(float)
    
    def softmax(self, z):
        """Softmax activation function for multi-class classification"""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        
        Args:
            X (np.array): Input data
            
        Returns:
            dict: Cache containing all intermediate values
        """
        cache = {'A0': X}
        
        for i in range(1, self.num_layers):
            # Linear transformation
            cache[f'Z{i}'] = np.dot(cache[f'A{i-1}'], self.weights[f'W{i}']) + self.biases[f'b{i}']
            
            # Apply activation function
            if i == self.num_layers - 1:  # Output layer
                cache[f'A{i}'] = self.softmax(cache[f'Z{i}'])
            else:  # Hidden layers
                cache[f'A{i}'] = self.relu(cache[f'Z{i}'])
        
        return cache
    
    def compute_cost(self, AL, Y):
        """
        Compute the cross-entropy cost.
        
        Args:
            AL (np.array): Predicted probabilities
            Y (np.array): True labels (one-hot encoded)
            
        Returns:
            float: Cost value
        """
        m = Y.shape[0]
        cost = -np.mean(np.sum(Y * np.log(AL + 1e-8), axis=1))
        return cost
    
    def backward_propagation(self, cache, Y):
        """
        Perform backward propagation to compute gradients.
        
        Args:
            cache (dict): Forward propagation cache
            Y (np.array): True labels (one-hot encoded)
            
        Returns:
            dict: Gradients for weights and biases
        """
        m = Y.shape[0]
        gradients = {}
        
        # Output layer gradients
        L = self.num_layers - 1
        dZ = cache[f'A{L}'] - Y
        gradients[f'dW{L}'] = (1/m) * np.dot(cache[f'A{L-1}'].T, dZ)
        gradients[f'db{L}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        
        # Hidden layer gradients
        for i in range(L-1, 0, -1):
            dA = np.dot(dZ, self.weights[f'W{i+1}'].T)
            dZ = dA * self.relu_derivative(cache[f'Z{i}'])
            gradients[f'dW{i}'] = (1/m) * np.dot(cache[f'A{i-1}'].T, dZ)
            gradients[f'db{i}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        
        return gradients
    
    def update_parameters(self, gradients):
        """Update weights and biases using gradients"""
        for i in range(1, self.num_layers):
            self.weights[f'W{i}'] -= self.learning_rate * gradients[f'dW{i}']
            self.biases[f'b{i}'] -= self.learning_rate * gradients[f'db{i}']
    
    def train(self, X_train, Y_train, X_val=None, Y_val=None, epochs=1000, print_cost=True, print_interval=100):
        """
        Train the neural network.
        
        Args:
            X_train (np.array): Training features
            Y_train (np.array): Training labels (one-hot encoded)
            X_val (np.array): Validation features (optional)
            Y_val (np.array): Validation labels (optional)
            epochs (int): Number of training epochs
            print_cost (bool): Whether to print cost during training
            print_interval (int): Interval for printing cost
        """
        for epoch in range(epochs):
            # Forward propagation
            cache = self.forward_propagation(X_train)
            
            # Compute cost
            cost = self.compute_cost(cache[f'A{self.num_layers-1}'], Y_train)
            self.cost_history.append(cost)
            
            # Backward propagation
            gradients = self.backward_propagation(cache, Y_train)
            
            # Update parameters
            self.update_parameters(gradients)
            
            # Compute accuracy
            if epoch % print_interval == 0 or epoch == epochs - 1:
                train_accuracy = self.accuracy(X_train, Y_train)
                self.accuracy_history.append(train_accuracy)
                
                if print_cost:
                    val_acc_str = ""
                    if X_val is not None and Y_val is not None:
                        val_accuracy = self.accuracy(X_val, Y_val)
                        val_acc_str = f", Val Accuracy: {val_accuracy:.4f}"
                    
                    print(f"Epoch {epoch}: Cost = {cost:.4f}, Train Accuracy = {train_accuracy:.4f}{val_acc_str}")
    
    def predict(self, X):
        """Make predictions on new data"""
        cache = self.forward_propagation(X)
        predictions = np.argmax(cache[f'A{self.num_layers-1}'], axis=1)
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        cache = self.forward_propagation(X)
        return cache[f'A{self.num_layers-1}']
    
    def accuracy(self, X, Y):
        """Compute accuracy on given data"""
        predictions = self.predict(X)
        true_labels = np.argmax(Y, axis=1)
        return np.mean(predictions == true_labels)
    
    def plot_training_history(self):
        """Plot training cost and accuracy history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot cost
        ax1.plot(self.cost_history)
        ax1.set_title('Training Cost')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cost')
        ax1.grid(True)
        
        # Plot accuracy
        epochs_acc = np.arange(0, len(self.cost_history), len(self.cost_history) // len(self.accuracy_history))[:len(self.accuracy_history)]
        ax2.plot(epochs_acc, self.accuracy_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'cost_history': self.cost_history,
            'accuracy_history': self.accuracy_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.layer_sizes = model_data['layer_sizes']
        self.learning_rate = model_data['learning_rate']
        self.cost_history = model_data['cost_history']
        self.accuracy_history = model_data['accuracy_history']
        self.num_layers = len(self.layer_sizes)
        print(f"Model loaded from {filepath}")


def load_mnist_data():
    """
    Load and preprocess MNIST dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - preprocessed data
    """
    print("Loading MNIST dataset...")
    
    # Load MNIST data from sklearn
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Convert labels to one-hot encoding
    y_one_hot = np.eye(10)[y]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Output classes: {y_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def visualize_samples(X, y, num_samples=10):
    """
    Visualize sample images from the dataset.
    
    Args:
        X (np.array): Image data
        y (np.array): Labels (one-hot encoded)
        num_samples (int): Number of samples to visualize
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    # Convert one-hot back to class labels
    y_labels = np.argmax(y, axis=1)
    
    for i in range(num_samples):
        # Reshape back to 28x28 image
        image = X[i].reshape(28, 28)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Label: {y_labels[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_predictions(model, X_test, y_test, num_samples=10):
    """
    Analyze model predictions with visualizations.
    
    Args:
        model: Trained neural network model
        X_test (np.array): Test features
        y_test (np.array): Test labels (one-hot encoded)
        num_samples (int): Number of samples to analyze
    """
    # Make predictions
    predictions = model.predict(X_test[:num_samples])
    probabilities = model.predict_proba(X_test[:num_samples])
    true_labels = np.argmax(y_test[:num_samples], axis=1)
    
    # Visualize predictions
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Display image
        image = X_test[i].reshape(28, 28)
        axes[i].imshow(image, cmap='gray')
        
        # Create title with prediction info
        confidence = probabilities[i][predictions[i]]
        color = 'green' if predictions[i] == true_labels[i] else 'red'
        
        title = f'True: {true_labels[i]}, Pred: {predictions[i]}\nConf: {confidence:.3f}'
        axes[i].set_title(title, color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\nPrediction Analysis:")
    print("-" * 50)
    for i in range(num_samples):
        status = "✓" if predictions[i] == true_labels[i] else "✗"
        confidence = probabilities[i][predictions[i]]
        print(f"Sample {i+1}: {status} True: {true_labels[i]}, Predicted: {predictions[i]}, Confidence: {confidence:.3f}")