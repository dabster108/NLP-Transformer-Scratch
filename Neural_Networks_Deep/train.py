# train.py - Training Script for Deep Neural Network
"""
Dedicated training script for the Deep Neural Network.
This script focuses on training the model with different configurations
and hyperparameter experimentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from neural import DeepNeuralNetwork, load_mnist_data
import time
import json

def experiment_with_architectures():
    """
    Experiment with different network architectures to find the best one.
    """
    print("🔬 ARCHITECTURE EXPERIMENTATION")
    print("=" * 50)
    
    # Load data
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # Use subset for faster experimentation
    subset_size = 5000
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    X_val = X_test[:1000]
    y_val = y_test[:1000]
    
    # Different architectures to test
    architectures = [
        [784, 64, 10],                    # Shallow network
        [784, 128, 64, 10],              # Medium network
        [784, 128, 64, 32, 10],          # Deep network
        [784, 256, 128, 64, 10],         # Wide network
        [784, 512, 256, 128, 64, 10],    # Very deep network
    ]
    
    results = []
    
    for i, arch in enumerate(architectures):
        print(f"\n🏗️  Testing Architecture {i+1}: {arch}")
        print("-" * 30)
        
        # Create and train model
        model = DeepNeuralNetwork(
            layer_sizes=arch,
            learning_rate=0.01,
            random_seed=42
        )
        
        start_time = time.time()
        model.train(
            X_train_subset, y_train_subset,
            X_val=X_val, Y_val=y_val,
            epochs=200,
            print_cost=False
        )
        training_time = time.time() - start_time
        
        # Evaluate
        train_acc = model.accuracy(X_train_subset, y_train_subset)
        val_acc = model.accuracy(X_val, y_val)
        
        results.append({
            'architecture': arch,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'training_time': training_time,
            'parameters': sum(arch[i] * arch[i+1] + arch[i+1] for i in range(len(arch)-1))
        })
        
        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {training_time:.2f}s")
    
    # Display results
    print("\n📊 ARCHITECTURE COMPARISON")
    print("=" * 80)
    print(f"{'#':<3} {'Architecture':<25} {'Train Acc':<10} {'Val Acc':<10} {'Params':<10} {'Time (s)':<8}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        arch_str = str(result['architecture'])
        print(f"{i+1:<3} {arch_str:<25} {result['train_accuracy']:<10.4f} "
              f"{result['val_accuracy']:<10.4f} {result['parameters']:<10,} {result['training_time']:<8.2f}")
    
    # Find best architecture
    best_arch = max(results, key=lambda x: x['val_accuracy'])
    print(f"\n🏆 Best Architecture: {best_arch['architecture']}")
    print(f"   Validation Accuracy: {best_arch['val_accuracy']:.4f}")
    
    return best_arch['architecture']


def experiment_with_learning_rates():
    """
    Experiment with different learning rates.
    """
    print("\n📈 LEARNING RATE EXPERIMENTATION")
    print("=" * 50)
    
    # Load data
    X_train, X_test, y_train, y_test = load_mnist_data()
    subset_size = 5000
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    
    # Different learning rates to test
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    architecture = [784, 128, 64, 32, 10]
    
    plt.figure(figsize=(15, 10))
    
    for i, lr in enumerate(learning_rates):
        print(f"🎯 Testing Learning Rate: {lr}")
        
        model = DeepNeuralNetwork(
            layer_sizes=architecture,
            learning_rate=lr,
            random_seed=42
        )
        
        model.train(
            X_train_subset, y_train_subset,
            epochs=300,
            print_cost=False
        )
        
        # Plot cost history
        plt.subplot(2, 3, i+1)
        plt.plot(model.cost_history)
        plt.title(f'LR = {lr}')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.grid(True)
        
        final_accuracy = model.accuracy(X_train_subset, y_train_subset)
        print(f"   Final Accuracy: {final_accuracy:.4f}")
    
    plt.subplot(2, 3, 6)
    # Plot comparison of all learning rates
    for i, lr in enumerate(learning_rates):
        model = DeepNeuralNetwork(layer_sizes=architecture, learning_rate=lr, random_seed=42)
        model.train(X_train_subset, y_train_subset, epochs=300, print_cost=False)
        plt.plot(model.cost_history, label=f'LR = {lr}')
    
    plt.title('Learning Rate Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def train_best_model():
    """
    Train the best model with optimal hyperparameters.
    """
    print("\n🚀 TRAINING BEST MODEL")
    print("=" * 50)
    
    # Load full dataset
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    # Use optimal architecture (you can modify based on experiments)
    best_architecture = [784, 128, 64, 32, 10]
    best_lr = 0.01
    
    print(f"Architecture: {best_architecture}")
    print(f"Learning Rate: {best_lr}")
    print(f"Training Samples: {len(X_train):,}")
    
    # Create model
    model = DeepNeuralNetwork(
        layer_sizes=best_architecture,
        learning_rate=best_lr,
        random_seed=42
    )
    
    # Train model
    print("\n🔄 Training in progress...")
    start_time = time.time()
    
    model.train(
        X_train, y_train,
        X_val=X_test[:2000],
        Y_val=y_test[:2000],
        epochs=1000,
        print_cost=True,
        print_interval=100
    )
    
    training_time = time.time() - start_time
    
    # Final evaluation
    train_accuracy = model.accuracy(X_train, y_train)
    test_accuracy = model.accuracy(X_test, y_test)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save the best model
    model.save_model("best_mnist_model.pkl")
    
    # Save training results
    results = {
        'architecture': best_architecture,
        'learning_rate': best_lr,
        'training_time': training_time,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'epochs': 1000
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 Model and results saved!")
    
    return model


def analyze_training_dynamics(model):
    """
    Analyze the training dynamics of the model.
    """
    print("\n🔍 TRAINING DYNAMICS ANALYSIS")
    print("=" * 50)
    
    # Plot detailed training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cost history
    axes[0, 0].plot(model.cost_history)
    axes[0, 0].set_title('Training Cost Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross-Entropy Loss')
    axes[0, 0].grid(True)
    
    # Accuracy history
    epochs_acc = np.arange(0, len(model.cost_history), 
                          len(model.cost_history) // len(model.accuracy_history))[:len(model.accuracy_history)]
    axes[0, 1].plot(epochs_acc, model.accuracy_history)
    axes[0, 1].set_title('Training Accuracy Over Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)
    
    # Cost change rate
    cost_changes = np.diff(model.cost_history)
    axes[1, 0].plot(cost_changes)
    axes[1, 0].set_title('Cost Change Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cost Change')
    axes[1, 0].grid(True)
    
    # Learning curve (log scale)
    axes[1, 1].semilogy(model.cost_history)
    axes[1, 1].set_title('Training Cost (Log Scale)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Log(Cost)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"📊 Training Statistics:")
    print(f"   Initial Cost: {model.cost_history[0]:.4f}")
    print(f"   Final Cost: {model.cost_history[-1]:.4f}")
    print(f"   Cost Reduction: {model.cost_history[0] - model.cost_history[-1]:.4f}")
    print(f"   Average Cost Change: {np.mean(cost_changes):.6f}")
    print(f"   Final Accuracy: {model.accuracy_history[-1]:.4f}")


if __name__ == "__main__":
    print("🎯 DEEP NEURAL NETWORK TRAINING EXPERIMENTS")
    print("=" * 60)
    
    # Run experiments
    print("Starting comprehensive training experiments...")
    
    # 1. Architecture experiments
    best_arch = experiment_with_architectures()
    
    # 2. Learning rate experiments
    experiment_with_learning_rates()
    
    # 3. Train best model
    best_model = train_best_model()
    
    # 4. Analyze training dynamics
    analyze_training_dynamics(best_model)
    
    print("\n🎉 ALL EXPERIMENTS COMPLETED!")
    print("Check the saved files:")
    print("  • best_mnist_model.pkl - Best trained model")
    print("  • training_results.json - Training statistics")