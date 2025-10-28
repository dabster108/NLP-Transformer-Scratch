# 🧠 Neural Networks from Scratch

A comprehensive collection of neural network implementations built from scratch using NumPy and PyTorch, demonstrating fundamental deep learning concepts through practical, real-world applications.

![Brain Neuron Structure](https://www.smartsheet.com/sites/default/files/IC-Brain-Neuron-Structure.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Projects](#projects)
  - [XOR Logic Gate Predictor](#1-xor-logic-gate-predictor)
  - [Neural Network Bank Dataset](#2-neural-network-bank-dataset)
  - [Sine Wave Explorer](#3-sine-wave-explorer)
  - [Text Processing with LSTM](#4-text-processing-with-lstm)
- [Key Concepts Demonstrated](#key-concepts-demonstrated)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Learning Path](#learning-path)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Overview

This repository serves as a **hands-on learning journey** through neural network fundamentals. Each project progressively introduces core concepts of deep learning, from basic logic gates to advanced sequence processing, all implemented without relying on high-level frameworks for the core algorithms.

**Why build from scratch?**
- 🎯 Understand the mathematics behind neural networks
- 🔍 Grasp forward and backward propagation intuitively
- 💡 Learn how activation functions, loss functions, and optimizers work
- 🚀 Build a strong foundation before using high-level frameworks

![Neural Networks in Data Mining](https://www.smartsheet.com/sites/default/files/IC-Neural-Networks-In-Data-Mining.svg)

---

## 🚀 Projects

### 1. **XOR Logic Gate Predictor**
**Difficulty:** Beginner  
**Type:** Classification (Non-linear Problem)

#### 🎯 Objective
Solve the classic XOR problem—a fundamental challenge that demonstrates why neural networks need hidden layers and non-linear activation functions.

#### 🧩 Problem Statement
The XOR (exclusive OR) gate outputs `1` only when exactly one input is `1`:

| Input A | Input B | XOR Output |
|---------|---------|------------|
| 0       | 0       | 0          |
| 0       | 1       | 1          |
| 1       | 0       | 1          |
| 1       | 1       | 0          |

**Why is this important?** XOR is not linearly separable, making it impossible for a single-layer perceptron to solve. This project shows why deep learning is necessary.

#### 🔧 Technical Implementation
- **Architecture:** 2-layer feedforward neural network (2 → 2 → 1)
- **Activation Functions:** ReLU (hidden layer), Sigmoid (output layer)
- **Loss Function:** Binary Cross-Entropy
- **Implementation:** Pure NumPy (from scratch)
- **Training:** Backpropagation with gradient descent

#### 📁 Files
```
XOR-Logic-Gate-Predictor/
├── neural.py    # Network architecture and training logic
├── trainxor.py  # Training script
├── main.py      # Testing and prediction
└── README.md    # Detailed documentation
```

#### 💡 Key Learnings
- Non-linear activation functions
- Backpropagation fundamentals
- Why depth matters in neural networks

---

### 2. **Neural Network Bank Dataset**
**Difficulty:** Intermediate  
**Type:** Binary Classification (Real-world Data)

#### 🎯 Objective
Predict whether a bank customer will subscribe to a term deposit based on demographic and campaign data.

#### 📊 Dataset
Uses the **Bank Marketing Dataset** (`bank.csv`) with features including:
- Age, job, marital status, education
- Account balance, loan status
- Campaign contact information
- Economic indicators

#### 🔧 Technical Implementation
- **Architecture:** Multi-layer perceptron with customizable hidden layers
- **Preprocessing:** Feature encoding, normalization, train-test split
- **Activation Functions:** ReLU (hidden), Sigmoid (output)
- **Loss Function:** Binary Cross-Entropy
- **Implementation:** NumPy-based with custom gradient computation
- **Features:**
  - Data cleaning and preprocessing pipeline
  - Categorical variable encoding
  - Feature scaling using StandardScaler
  - Custom prediction function

#### 📁 Files
```
Neural_Netowrk_Bank_Dataset/
├── bank.csv           # Raw dataset
├── data_clean.py      # Preprocessing pipeline
├── neural.py          # Network implementation
├── train.py           # Training script
├── main.py            # Prediction and evaluation
└── neural_networks_.ipynb  # Interactive notebook
```

#### 💡 Key Learnings
- Data preprocessing for neural networks
- Handling categorical and numerical features
- Real-world classification problems
- Model evaluation metrics

---

### 3. **Sine Wave Explorer** 🌊
**Difficulty:** Intermediate-Advanced  
**Type:** Multi-task Learning (Regression + Classification)

#### 🎯 Objective
A dual-purpose neural network that performs:
1. **Regression:** Predict `sin(x)` values for any input `x`
2. **Classification:** Identify sine wave frequency (1Hz, 2Hz, or 3Hz)

#### 🔧 Technical Implementation
- **Architecture:** Multi-task neural network with shared layers
  - Regression branch: 1 → 32 → 64 → 32 → 1
  - Classification branch: 20 → 32 → 64 → 32 → 3
- **Framework:** PyTorch
- **Features:**
  - Shared hidden representations for both tasks
  - Task-specific input layers
  - Dual output heads (regression + classification)
  - Model persistence (saves/loads trained weights)

#### 📊 Functionality
- **Sine Prediction:** Takes single `x` value, predicts `sin(x)`
- **Frequency Recognition:** Takes 20 sampled points, classifies wave frequency
- **Visualization:** Plots predicted vs actual sine curves

#### 📁 Files
```
Sine_Wave_Explorer/
├── neural.py          # Multi-task network architecture
├── train.py           # Training for both tasks
├── main.py            # Testing and visualization
├── sinewave_model.pt  # Saved model weights
└── README.md          # Documentation
```

#### 💡 Key Learnings
- Multi-task learning architecture
- PyTorch model building
- Signal processing with neural networks
- Shared vs. task-specific layers
- Model serialization

---

### 4. **Text Processing with LSTM** 📝
**Difficulty:** Advanced  
**Type:** Sequence Processing (Natural Language Processing)

#### 🎯 Objective
Process and understand text sequences using LSTM (Long Short-Term Memory) networks—a fundamental architecture for NLP tasks.

#### 🔧 Technical Implementation
- **Architecture:** LSTM-based sequence model
  - Embedding layer (converts words to vectors)
  - LSTM layer (processes sequences)
  - Fully connected output layer
- **Framework:** PyTorch
- **NLP Pipeline:**
  - Text cleaning (lowercase, remove punctuation/digits)
  - Tokenization
  - Vocabulary building
  - Sequence padding
  - Word-to-index mapping

#### 📊 Components
1. **Text Preprocessing:**
   - Cleaning and normalization
   - Tokenization
   - Vocabulary creation
   - Sequence padding to fixed length

2. **Model Architecture:**
   ```
   Embedding → LSTM → Fully Connected → Output
   ```

3. **Features:**
   - Custom vocabulary builder
   - Dynamic sequence padding
   - PyTorch Dataset and DataLoader integration
   - Probability-based predictions

#### 📁 Files
```
Text_Processing/
├── main.py    # Complete NLP pipeline and LSTM model
└── test.txt   # Sample text data
```

#### 💡 Key Learnings
- NLP preprocessing techniques
- Word embeddings
- LSTM architecture and applications
- Sequence padding and batching
- PyTorch Dataset/DataLoader patterns

---

## 🎓 Key Concepts Demonstrated

### Neural Network Fundamentals
- ✅ **Forward Propagation:** Computing predictions through layers
- ✅ **Backward Propagation:** Computing gradients for weight updates
- ✅ **Activation Functions:** ReLU, Sigmoid, and their derivatives
- ✅ **Loss Functions:** Binary Cross-Entropy, MSE
- ✅ **Optimization:** Gradient descent

### Advanced Techniques
- 🔄 **Multi-task Learning:** Single network, multiple objectives
- 📊 **Data Preprocessing:** Normalization, encoding, feature engineering
- 🧠 **Sequence Processing:** LSTMs for temporal/sequential data
- 💾 **Model Persistence:** Saving and loading trained models
- 📈 **Visualization:** Matplotlib for training curves and predictions

### Mathematical Foundations
- Linear algebra operations (matrix multiplication, dot products)
- Calculus (derivatives, chain rule for backpropagation)
- Probability and statistics (loss functions, predictions)

![Simplified Artificial Neural Networks](https://www.smartsheet.com/sites/default/files/IC-simplified-artificial-neural-networks-corrected.svg)

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Primary programming language |
| **NumPy** | Matrix operations, numerical computation |
| **PyTorch** | Deep learning framework (Sine Wave, Text Processing) |
| **Pandas** | Data manipulation and preprocessing |
| **Matplotlib** | Data visualization and results plotting |
| **Scikit-learn** | Preprocessing utilities (scaling, encoding) |

---

## 🚀 Getting Started

### Prerequisites
```bash
python --version  # Python 3.7 or higher
```

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/dabster108/Neural-Networks-Scratch.git
   cd Neural-Networks-Scratch
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib torch scikit-learn
   ```

### Running Projects

#### XOR Logic Gate
```bash
cd "XOR-Logic-Gate-Predictor "
python trainxor.py  # Train the model
python main.py      # Test predictions
```

#### Bank Dataset Classification
```bash
cd Neural_Netowrk_Bank_Dataset
python main.py      # Trains and makes predictions
```

#### Sine Wave Explorer
```bash
cd Sine_Wave_Explorer
python train.py     # Train the model
python main.py      # Visualize predictions
```

#### Text Processing
```bash
cd Text_Processing
python main.py      # Run NLP pipeline
```

---

## 📚 Learning Path

**Recommended Order for Beginners:**

1. **Start Here:** XOR Logic Gate Predictor  
   *Learn the basics of backpropagation and non-linear problems*

2. **Next:** Neural Network Bank Dataset  
   *Apply concepts to real-world data with preprocessing*

3. **Then:** Sine Wave Explorer  
   *Explore PyTorch and multi-task learning*

4. **Advanced:** Text Processing with LSTM  
   *Dive into sequence models and NLP*

---

## 📁 Project Structure

```
Neural-Networks-Scratch/
│
├── XOR-Logic-Gate-Predictor/       # Basic classification problem
│   ├── neural.py
│   ├── trainxor.py
│   ├── main.py
│   └── README.md
│
├── Neural_Netowrk_Bank_Dataset/    # Real-world binary classification
│   ├── bank.csv
│   ├── data_clean.py
│   ├── neural.py
│   ├── train.py
│   ├── main.py
│   └── neural_networks_.ipynb
│
├── Sine_Wave_Explorer/             # Multi-task learning (PyTorch)
│   ├── neural.py
│   ├── train.py
│   ├── main.py
│   ├── sinewave_model.pt
│   └── README.md
│
├── Text_Processing/                # NLP with LSTM (PyTorch)
│   ├── main.py
│   └── test.txt
│
└── README.md                       # This file
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:
- Add more activation functions (Tanh, Leaky ReLU)
- Implement additional optimizers (Adam, RMSprop)
- Add visualization for decision boundaries
- Create new projects (CNNs, GANs, Autoencoders)
- Improve documentation

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📖 Additional Resources

- [Neural Networks and Deep Learning (Free Book)](http://neuralnetworksanddeeplearning.com/)
- [3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NumPy for Neural Networks](https://numpy.org/doc/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Dikshanta**  
GitHub: [@dabster108](https://github.com/dabster108)

---

## ⭐ Acknowledgments

- Inspired by Andrew Ng's Deep Learning Specialization
- Neural network visualizations from Smartsheet
- XOR problem: Classic AI challenge from the 1960s

---

**Happy Learning! 🚀**  
*"The best way to learn is by building."*
