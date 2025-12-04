# Deep Neural Network for MNIST Classification

## 🎯 Project Overview

This project implements a **Deep Neural Network from scratch** using only NumPy to classify handwritten digits from the MNIST dataset. It demonstrates fundamental deep learning concepts including multi-layer perceptrons, backpropagation, and advanced training techniques.

### Why This Project?
- 🧠 **Understand Deep Learning Fundamentals**: Build neural networks without high-level frameworks
- 📊 **Real Dataset**: Work with the famous MNIST handwritten digit dataset (70,000 images)
- 🔬 **Experiment with Architecture**: Test different network depths and widths
- 📈 **Performance Analysis**: Achieve high accuracy through proper design
- 💾 **Model Persistence**: Save and load trained models

---

## 🌟 Key Features

### Deep Neural Network Implementation
- **Multi-layer Architecture**: Support for arbitrary number of hidden layers
- **Xavier Initialization**: Smart weight initialization for better convergence
- **Activation Functions**: ReLU for hidden layers, Softmax for output
- **Loss Function**: Cross-entropy loss for multi-class classification
- **Optimization**: Gradient descent with backpropagation

### Advanced Training Features
- **Training Visualization**: Real-time cost and accuracy plotting
- **Model Persistence**: Save/load trained models
- **Hyperparameter Experiments**: Test different architectures and learning rates
- **Comprehensive Analysis**: Detailed prediction analysis with confidence scores

### Interactive Components
- **Visual Predictions**: See what the model predicts vs actual labels
- **Interactive Testing**: Test the model on specific samples
- **Training Dynamics**: Analyze how the model learns over time

---

## 📊 Dataset: MNIST Handwritten Digits

- **Size**: 70,000 images (60,000 training + 10,000 testing)
- **Image Size**: 28×28 pixels (784 features when flattened)
- **Classes**: 10 digits (0-9)
- **Preprocessing**: Normalized pixel values (0-1), one-hot encoded labels

---

## 🏗️ Architecture

### Default Network Structure
```
Input Layer:    784 neurons (28×28 pixels)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 64 neurons  (ReLU activation)  
Hidden Layer 3: 32 neurons  (ReLU activation)
Output Layer:   10 neurons  (Softmax activation)

Total Parameters: ~135,000
```

### Mathematical Foundation
```
Forward Pass:
Z[l] = W[l] × A[l-1] + b[l]
A[l] = activation(Z[l])

Backward Pass:
dW[l] = (1/m) × A[l-1]T × dZ[l]
db[l] = (1/m) × sum(dZ[l])
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn
```

### Quick Start
```bash
# Run the complete training and analysis
python main.py

# Run architecture experiments
python train.py

# Interactive prediction mode (after training)
python main.py  # then choose 'y' for interactive mode
```

---

## 📁 File Structure

```
Neural_Networks_Deep/
├── neural.py          # Core neural network implementation
├── main.py            # Main training and evaluation script
├── train.py           # Advanced training experiments
├── README.md          # This documentation
├── best_mnist_model.pkl      # Saved trained model (after training)
└── training_results.json     # Training statistics (after training)
```

---

## 🔧 Core Components

### 1. `neural.py` - Neural Network Engine
```python
class DeepNeuralNetwork:
    - Multi-layer architecture support
    - Xavier weight initialization
    - Forward/backward propagation
    - Training with validation
    - Model persistence
```

### 2. `main.py` - Complete Training Pipeline
- **Step 1**: Load and visualize MNIST data
- **Step 2**: Design network architecture
- **Step 3**: Train the neural network
- **Step 4**: Evaluate performance
- **Step 5**: Visualize training progress
- **Step 6**: Analyze predictions
- **Step 7**: Save trained model
- **Step 8**: Interactive prediction mode

### 3. `train.py` - Advanced Experiments
- Architecture comparison experiments
- Learning rate optimization
- Training dynamics analysis
- Hyperparameter tuning

---

## 📈 Expected Results

### Performance Metrics
- **Training Accuracy**: ~95-98%
- **Test Accuracy**: ~92-95%
- **Training Time**: 2-5 minutes (depending on hardware)
- **Convergence**: Usually within 500 epochs

### Training Visualization
The model provides comprehensive training visualizations:
- Cost function over time
- Accuracy progression
- Prediction confidence analysis
- Sample predictions with probabilities

---

## 🧪 Experiments You Can Try

### 1. Architecture Experiments
```python
# Test different architectures
architectures = [
    [784, 64, 10],                    # Shallow
    [784, 128, 64, 10],              # Medium  
    [784, 128, 64, 32, 10],          # Deep
    [784, 256, 128, 64, 10],         # Wide
]
```

### 2. Learning Rate Tuning
```python
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
```

### 3. Advanced Features to Add
- **Regularization**: L1/L2 penalty, Dropout
- **Optimization**: Adam, RMSprop optimizers
- **Normalization**: Batch normalization
- **Advanced Activations**: Leaky ReLU, Swish
- **Learning Rate Scheduling**: Exponential decay

---

## 🎓 Learning Objectives

After completing this project, you'll understand:

### Mathematical Concepts
- ✅ **Matrix Operations**: Forward propagation mathematics
- ✅ **Calculus**: Backpropagation and chain rule
- ✅ **Probability**: Softmax and cross-entropy loss
- ✅ **Optimization**: Gradient descent principles

### Programming Skills
- ✅ **NumPy Mastery**: Advanced array operations
- ✅ **Object-Oriented Design**: Clean class structure
- ✅ **Data Visualization**: Matplotlib for analysis
- ✅ **Model Persistence**: Saving/loading models

### Deep Learning Fundamentals
- ✅ **Network Architecture**: Layer design principles
- ✅ **Training Dynamics**: How networks learn
- ✅ **Evaluation Metrics**: Accuracy, loss analysis
- ✅ **Overfitting**: Training vs validation performance

---

## 🔍 Key Code Snippets

### Making Predictions
```python
# Load trained model
model = DeepNeuralNetwork(layer_sizes=[784, 128, 64, 32, 10])
model.load_model("best_mnist_model.pkl")

# Predict on new data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Custom Training
```python
# Create and train model
model = DeepNeuralNetwork(
    layer_sizes=[784, 128, 64, 32, 10],
    learning_rate=0.01
)

model.train(X_train, y_train, epochs=500)
```

---

## 🚀 Next Steps

### Immediate Improvements
1. **Add Regularization**: Implement dropout and L2 regularization
2. **Better Optimizers**: Add Adam and RMSprop
3. **Data Augmentation**: Rotate/scale images for better generalization
4. **Batch Processing**: Implement mini-batch gradient descent

### Advanced Extensions
1. **Convolutional Layers**: Add CNN layers for better image processing
2. **Transfer Learning**: Use pre-trained features
3. **Ensemble Methods**: Combine multiple models
4. **Hyperparameter Search**: Automated optimization

---

## 📚 Additional Resources

- [Neural Networks and Deep Learning (Free Book)](http://neuralnetworksanddeeplearning.com/)
- [3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Backpropagation Explained](https://brilliant.org/wiki/backpropagation/)

---

## 🤝 Contributing

Ideas for contributions:
- Add different activation functions
- Implement advanced optimizers
- Create visualization tools
- Add regularization techniques
- Improve documentation

---

## 👤 Author

**Dikshanta**  
GitHub: [@dabster108](https://github.com/dabster108)

---

**Happy Deep Learning! 🧠✨**