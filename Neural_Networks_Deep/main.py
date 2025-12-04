# main.py - Deep Neural Network for MNIST Digit Classification
"""
Deep Neural Network Implementation from Scratch

This project demonstrates building a deep neural network from scratch using only NumPy
to classify handwritten digits from the MNIST dataset. The network supports multiple
hidden layers and showcases fundamental deep learning concepts.

Key Features:
- Multi-layer perceptron with customizable architecture
- Xavier weight initialization for better convergence
- ReLU activation for hidden layers, Softmax for output
- Cross-entropy loss with backpropagation
- Training visualization and model persistence
- Comprehensive prediction analysis

Author: Dikshanta
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from neural import DeepNeuralNetwork, load_mnist_data, visualize_samples, analyze_predictions
import time
import os

def main():
    """Main function to train and test the deep neural network"""
    
    print("🧠 Deep Neural Network for MNIST Digit Classification")
    print("=" * 60)
    print("Building a neural network from scratch using NumPy!")
    print()
    
    # =============================================
    # Step 1: Load and Explore the Dataset
    # =============================================
    print("📊 STEP 1: Loading MNIST Dataset")
    print("-" * 40)
    
    try:
        X_train, X_test, y_train, y_test = load_mnist_data()
        print("✅ Dataset loaded successfully!")
        print()
        
        # Visualize some sample images
        print("🖼️  Sample images from the dataset:")
        visualize_samples(X_train, y_train, num_samples=10)
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure you have internet connection for downloading MNIST dataset.")
        return
    
    # =============================================
    # Step 2: Design Network Architecture
    # =============================================
    print("🏗️  STEP 2: Designing Network Architecture")
    print("-" * 40)
    
    # Define network architecture
    input_size = 784    # 28x28 pixels flattened
    hidden1_size = 128  # First hidden layer
    hidden2_size = 64   # Second hidden layer
    hidden3_size = 32   # Third hidden layer
    output_size = 10    # 10 digit classes (0-9)
    
    layer_sizes = [input_size, hidden1_size, hidden2_size, hidden3_size, output_size]
    
    print(f"Network Architecture:")
    print(f"  Input Layer:     {input_size} neurons (28x28 pixels)")
    print(f"  Hidden Layer 1:  {hidden1_size} neurons (ReLU)")
    print(f"  Hidden Layer 2:  {hidden2_size} neurons (ReLU)")
    print(f"  Hidden Layer 3:  {hidden3_size} neurons (ReLU)")
    print(f"  Output Layer:    {output_size} neurons (Softmax)")
    print(f"  Total Parameters: {calculate_parameters(layer_sizes):,}")
    print()
    
    # =============================================
    # Step 3: Initialize and Train the Model
    # =============================================
    print("🚀 STEP 3: Training the Neural Network")
    print("-" * 40)
    
    # Create the model
    model = DeepNeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.01,
        random_seed=42
    )
    
    # Training parameters
    epochs = 500
    print_interval = 50
    
    print(f"Training Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {model.learning_rate}")
    print(f"  Batch Size: Full batch")
    print(f"  Optimizer: Gradient Descent")
    print()
    
    # Start training
    print("🔄 Training in progress...")
    start_time = time.time()
    
    # Use a subset for faster training (you can use full dataset for better results)
    train_subset = 10000  # Use first 10k samples for demo
    X_train_subset = X_train[:train_subset]
    y_train_subset = y_train[:train_subset]
    
    # Train the model
    model.train(
        X_train_subset, y_train_subset,
        X_val=X_test[:1000],  # Use first 1k test samples for validation
        Y_val=y_test[:1000],
        epochs=epochs,
        print_cost=True,
        print_interval=print_interval
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds!")
    print()
    
    # =============================================
    # Step 4: Evaluate Model Performance
    # =============================================
    print("📈 STEP 4: Model Evaluation")
    print("-" * 40)
    
    # Calculate accuracies
    train_accuracy = model.accuracy(X_train_subset, y_train_subset)
    test_accuracy = model.accuracy(X_test, y_test)
    
    print(f"📊 Performance Metrics:")
    print(f"  Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Training Samples:    {train_subset:,}")
    print(f"  Test Samples:        {len(X_test):,}")
    print()
    
    # =============================================
    # Step 5: Visualize Training Progress
    # =============================================
    print("📉 STEP 5: Training Visualization")
    print("-" * 40)
    
    print("🎯 Displaying training history...")
    model.plot_training_history()
    
    # =============================================
    # Step 6: Analyze Predictions
    # =============================================
    print("🔍 STEP 6: Prediction Analysis")
    print("-" * 40)
    
    print("🖼️  Analyzing model predictions on test samples...")
    analyze_predictions(model, X_test, y_test, num_samples=10)
    
    # =============================================
    # Step 7: Save the Model
    # =============================================
    print("💾 STEP 7: Saving the Model")
    print("-" * 40)
    
    model_path = "mnist_deep_model.pkl"
    model.save_model(model_path)
    print(f"✅ Model saved as '{model_path}'")
    print()
    
    # =============================================
    # Step 8: Demonstrate Model Loading
    # =============================================
    print("🔄 STEP 8: Model Loading Demo")
    print("-" * 40)
    
    # Create a new model and load the saved weights
    new_model = DeepNeuralNetwork(layer_sizes=layer_sizes)
    new_model.load_model(model_path)
    
    # Verify the loaded model works
    loaded_accuracy = new_model.accuracy(X_test[:1000], y_test[:1000])
    print(f"✅ Loaded model accuracy: {loaded_accuracy:.4f}")
    print()
    
    # =============================================
    # Summary and Next Steps
    # =============================================
    print("🎉 PROJECT SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully built and trained a {len(layer_sizes)}-layer neural network")
    print(f"✅ Achieved {test_accuracy*100:.2f}% accuracy on MNIST digit classification")
    print(f"✅ Implemented from scratch using only NumPy")
    print(f"✅ Training time: {training_time:.2f} seconds")
    print()
    
    print("🚀 NEXT STEPS TO IMPROVE:")
    print("1. 🔧 Add different optimizers (Adam, RMSprop)")
    print("2. 📏 Implement regularization (L2, Dropout)")
    print("3. 🎯 Add batch normalization")
    print("4. 📊 Try different activation functions")
    print("5. 🎨 Add convolutional layers for better image processing")
    print("6. 📈 Implement learning rate scheduling")
    print()
    
    print("💡 Key Concepts Demonstrated:")
    print("• Forward propagation through multiple layers")
    print("• Backpropagation with chain rule")
    print("• Xavier weight initialization")
    print("• Softmax activation for multi-class classification")
    print("• Cross-entropy loss function")
    print("• Model persistence and loading")
    print()


def calculate_parameters(layer_sizes):
    """Calculate total number of parameters in the network"""
    total_params = 0
    for i in range(1, len(layer_sizes)):
        # Weights + biases
        total_params += layer_sizes[i-1] * layer_sizes[i] + layer_sizes[i]
    return total_params


def interactive_prediction():
    """Interactive function to test the model on specific digits"""
    print("\n🎮 INTERACTIVE PREDICTION MODE")
    print("-" * 40)
    
    # Load the saved model
    try:
        model = DeepNeuralNetwork(layer_sizes=[784, 128, 64, 32, 10])
        model.load_model("mnist_deep_model.pkl")
        
        # Load test data
        _, X_test, _, y_test = load_mnist_data()
        
        while True:
            try:
                idx = input("\nEnter test sample index (0-13999) or 'q' to quit: ")
                if idx.lower() == 'q':
                    break
                
                idx = int(idx)
                if 0 <= idx < len(X_test):
                    # Make prediction
                    sample = X_test[idx:idx+1]
                    prediction = model.predict(sample)[0]
                    probabilities = model.predict_proba(sample)[0]
                    true_label = np.argmax(y_test[idx])
                    
                    # Display image
                    plt.figure(figsize=(6, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(sample.reshape(28, 28), cmap='gray')
                    plt.title(f'True: {true_label}, Predicted: {prediction}')
                    plt.axis('off')
                    
                    # Display probability distribution
                    plt.subplot(1, 2, 2)
                    plt.bar(range(10), probabilities)
                    plt.title('Prediction Probabilities')
                    plt.xlabel('Digit')
                    plt.ylabel('Probability')
                    plt.xticks(range(10))
                    
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"Prediction: {prediction} (Confidence: {probabilities[prediction]:.3f})")
                    
                else:
                    print("Index out of range! Please enter a number between 0 and 13999.")
                    
            except ValueError:
                print("Invalid input! Please enter a number or 'q' to quit.")
            except KeyboardInterrupt:
                break
        
        print("👋 Thanks for testing the model!")
        
    except FileNotFoundError:
        print("❌ Model file not found! Please run the main training first.")


if __name__ == "__main__":
    # Run the main training and evaluation
    main()
    
    # Ask if user wants to try interactive prediction
    while True:
        try:
            choice = input("\n🎮 Would you like to try interactive prediction? (y/n): ").lower()
            if choice == 'y':
                interactive_prediction()
                break
            elif choice == 'n':
                print("👋 Goodbye! Thanks for exploring deep learning!")
                break
            else:
                print("Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
