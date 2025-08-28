import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_fashion_mnist
from neural_network import NeuralNetwork
from optimizers import GradientDescent, Adam
from newtons_method import EfficientNewtonsMethod
from train import train_model
from utils import (plot_comparison_figure, generate_dynamic_report, 
                  plot_newton_comparison, generate_newton_report, save_plots)

def run_gd_vs_adam_comparison():
    """Run comparison between Gradient Descent and Adam with larger dataset"""
    print("=" * 60)
    print("GD vs ADAM COMPARISON (Larger Dataset)")
    print("=" * 60)
    
    # Load larger dataset for GD vs Adam
    X_train, X_test, y_train, y_test = load_fashion_mnist(subset_size=40000, test_size=0.2)
    
    # Split training data into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Dataset sizes for GD vs Adam comparison:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples")
    
    # Define network architecture
    input_size = X_train.shape[1]
    hidden_size1 = 128
    hidden_size2 = 64
    hidden_size3 = 32
    output_size = 10
    
    layer_sizes = [input_size, hidden_size1, hidden_size2, hidden_size3, output_size]
    
    # Define hyperparameters
    epochs = 30
    batch_size = 64
    dropout_rate = 0.3
    l2_lambda = 0.001
    
    # Train with Gradient Descent
    print("\n" + "="*50)
    print("TRAINING WITH GRADIENT DESCENT")
    print("="*50)
    
    nn_gd = NeuralNetwork(layer_sizes, 
                         dropout_rates=[dropout_rate, dropout_rate, dropout_rate, 0.0],
                         l2_lambda=l2_lambda)
    
    gd_optimizer = GradientDescent(learning_rate=0.02)
    history_gd = train_model(nn_gd, gd_optimizer, X_train, y_train, X_val, y_val, 
                            epochs=epochs, batch_size=batch_size)
    
    # Train with Adam
    print("\n" + "="*50)
    print("TRAINING WITH ADAM")
    print("="*50)
    
    nn_adam = NeuralNetwork(layer_sizes, 
                           dropout_rates=[dropout_rate, dropout_rate, dropout_rate, 0.0],
                           l2_lambda=l2_lambda)
    
    adam_optimizer = Adam(learning_rate=0.001)
    history_adam = train_model(nn_adam, adam_optimizer, X_train, y_train, X_val, y_val, 
                              epochs=epochs, batch_size=batch_size)
    
    # Evaluate on test set
    y_test_pred_gd = nn_gd.forward(X_test, training=False)
    test_accuracy_gd = nn_gd.compute_accuracy(y_test, y_test_pred_gd)
    
    y_test_pred_adam = nn_adam.forward(X_test, training=False)
    test_accuracy_adam = nn_adam.compute_accuracy(y_test, y_test_pred_adam)
    
    # Plot results
    fig_gd_adam = plot_comparison_figure(history_gd, history_adam)
    
    # Generate report
    generate_dynamic_report(history_gd, history_adam, test_accuracy_gd, test_accuracy_adam)
    
    return history_gd, history_adam, test_accuracy_gd, test_accuracy_adam, fig_gd_adam

def run_adam_vs_newton_comparison():
    """Run comparison between Adam and Newton's Method with smaller dataset"""
    print("\n" + "=" * 60)
    print("ADAM vs NEWTON COMPARISON (Smaller Dataset)")
    print("=" * 60)
    
    # Load smaller dataset for Newton's Method
    X_train, X_test, y_train, y_test = load_fashion_mnist(subset_size=1000, test_size=0.2)
    
    # Split training data into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Dataset sizes for Adam vs Newton comparison:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples")
    
    # Define smaller network architecture for Newton's Method
    input_size = X_train.shape[1]
    hidden_size1 = 64
    hidden_size2 = 32
    hidden_size3 = 16
    output_size = 10
    
    layer_sizes = [input_size, hidden_size1, hidden_size2, hidden_size3, output_size]
    
    # Define hyperparameters
    epochs = 10
    batch_size = 32
    
    # Train with Adam
    print("\n" + "="*50)
    print("TRAINING WITH ADAM (For Newton Comparison)")
    print("="*50)
    
    nn_adam = NeuralNetwork(layer_sizes, 
                           dropout_rates=[0.2, 0.2, 0.2, 0.0],
                           l2_lambda=0.001)
    
    adam_optimizer = Adam(learning_rate=0.001)
    history_adam_newton = train_model(nn_adam, adam_optimizer, X_train, y_train, X_val, y_val, 
                                    epochs=epochs, batch_size=batch_size)
    
    # Train with Newton's Method
    print("\n" + "="*50)
    print("TRAINING WITH NEWTON'S METHOD")
    print("="*50)
    print("Note: Newton's Method is computationally expensive and will be slow.")
    
    nn_newton = NeuralNetwork(layer_sizes, 
                             dropout_rates=[0.2, 0.2, 0.2, 0.0],
                             l2_lambda=0.001)
    
    newton_optimizer = EfficientNewtonsMethod(learning_rate=0.01, epsilon=1e-6, h=1e-4)
    history_newton = train_model(nn_newton, newton_optimizer, X_train, y_train, X_val, y_val, 
                               epochs=epochs, batch_size=batch_size)
    
    # Evaluate on test set
    y_test_pred_adam = nn_adam.forward(X_test, training=False)
    test_accuracy_adam_newton = nn_adam.compute_accuracy(y_test, y_test_pred_adam)
    
    y_test_pred_newton = nn_newton.forward(X_test, training=False)
    test_accuracy_newton = nn_newton.compute_accuracy(y_test, y_test_pred_newton)
    
    # Plot results
    fig_adam_newton = plot_newton_comparison(history_adam_newton, history_newton)
    
    # Generate report
    generate_newton_report(history_adam_newton, history_newton, test_accuracy_adam_newton, test_accuracy_newton)
    
    return history_adam_newton, history_newton, test_accuracy_adam_newton, test_accuracy_newton, fig_adam_newton

def main():
    """Run both comparisons"""
    print("COMPREHENSIVE OPTIMIZER COMPARISON STUDY")
    print("=" * 50)
    
    # Run GD vs Adam comparison
    gd_history, adam_history, gd_test_acc, adam_test_acc, fig_gd_adam = run_gd_vs_adam_comparison()
    
    # Run Adam vs Newton comparison
    adam_newton_history, newton_history, adam_newton_test_acc, newton_test_acc, fig_adam_newton = run_adam_vs_newton_comparison()
    
    # Save plots
    save_plots(fig_gd_adam, fig_adam_newton)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: ALL OPTIMIZERS COMPARISON")
    print("=" * 70)
    
    print(f"\nTest Accuracy Results:")
    print(f"  Gradient Descent: {gd_test_acc:.4f}")
    print(f"  Adam: {adam_test_acc:.4f}")
    print(f"  Newton's Method: {newton_test_acc:.4f}")
    
    print(f"\nPerformance Ranking:")
    optimizers = [
        ("Adam", adam_test_acc),
        ("Gradient Descent", gd_test_acc),
        ("Newton's Method", newton_test_acc)
    ]
    optimizers.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, acc) in enumerate(optimizers, 1):
        print(f"  {i}. {name}: {acc:.4f}")
    
    print(f"\nKey Insights:")
    print("1. Adam generally performs best for neural network optimization")
    print("2. Gradient Descent is reliable but slower to converge")
    print("3. Newton's Method is computationally expensive and often unstable for neural networks")
    print("4. Adaptive methods like Adam scale better with larger datasets")
    
    print(f"\nRecommendations:")
    print("✓ Use Adam for most neural network training tasks")
    print("✓ Consider Gradient Descent with momentum for some problems")
    print("✓ Use Newton's Method only for small, well-conditioned problems")
    print("✓ Always monitor validation performance during training")

if __name__ == "__main__":
    main()