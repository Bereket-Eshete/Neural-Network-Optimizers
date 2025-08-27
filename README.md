# Neural Network Optimizers Comparison Study

## 📋 Project Overview

This project implements and compares three different optimization algorithms for neural networks from scratch using only NumPy. The study provides practical insights into optimizer performance, convergence behavior, and computational efficiency.

## 🎯 Objectives

- Implement neural network with 3 hidden layers using only NumPy
- Compare Gradient Descent vs Adam optimization
- Implement Newton's Method with numerical second derivatives (bonus)
- Analyze performance differences and computational requirements
- Provide visual comparisons and comprehensive reports

## 🏗️ Architecture

The neural network architecture includes:

- **Input layer**: 784 features (Fashion-MNIST images)
- **Hidden layers**: 128 → 64 → 32 neurons with ReLU activation
- **Output layer**: 10 neurons with Softmax activation (multi-class classification)
- **Regularization**: L2 regularization and Dropout applied

## ⚙️ Optimizers Implemented

### 1. Mini-Batch Gradient Descent

python
w = w - η \* ∇L(w)

· Fixed learning rate
· Stable but slower convergence
· Simple implementation

2. Adam (Adaptive Moment Estimation)

python
m*t = β1\*m*{t-1} + (1-β1)*g_t
v_t = β2*v\_{t-1} + (1-β2)_g_t²
w = w - η _ m_hat / (√v_hat + ε)

· Adaptive learning rates per parameter
· Momentum for faster convergence
· Best overall performance

3. Newton's Method (Numerical Approximation)

python
w = w - η _ H⁻¹ _ ∇L(w)

· Uses second derivatives (curvature information)
· Computationally expensive
· Numerical instability issues

📊 Dataset

Fashion-MNIST - 70,000 grayscale images of 10 fashion categories:

· T-shirt/top, Trouser, Pullover, Dress, Coat
· Sandal, Shirt, Sneaker, Bag, Ankle boot

Dataset Sizes:

· GD vs Adam: 10,000 samples
· Adam vs Newton: 1,000 samples (due to computational constraints)

🚀 Implementation Highlights

Core Features:

· ✅ Neural network built from scratch with NumPy \
· ✅ 3 hidden layers with ReLU activation \
· ✅ Mini-batch training implementation \
· ✅ L2 regularization and Dropout \
· ✅ Comprehensive logging and visualization \
· ✅ Performance comparison reports \

Technical Stack:

· Python 3.x
· NumPy (core computations)
· Scikit-learn (data loading and preprocessing)
· Matplotlib (visualization)

📈 Results Summary

Performance Ranking:

1. Adam: 84.15% test accuracy ✅
2. Gradient Descent: 83.25% test accuracy ✅
3. Newton's Method: 19.50% test accuracy ❌

Key Findings:

Adam Advantages:

· Fastest convergence (18.1% improvement from baseline)
· Best final performance (84.15% accuracy)
· Adaptive learning rates handle different parameter sensitivities

Gradient Descent:

· Reliable and stable convergence
· Simpler implementation
· Good performance (83.25% accuracy)

Newton's Method Limitations:

· Numerical instability in finite precision
· Extremely computationally expensive (1.5+ seconds per epoch)
· Failed to learn effectively (19.50% accuracy ≈ random guessing)

📁 Project Structure

neural-network-optimizer/
├── data_loader.py # Dataset loading and preprocessing \
├── neural_network.py # Neural network architecture \
├── layers.py # Dense layer implementation \
├── optimizers.py # GD and Adam optimizers \
├── newton_method.py # Newton's Method implementation \
├── train.py # Training loop and logic \
├── utils.py # Visualization and reporting \
├── comparison_runner.py # Main comparison script \
├── gd_vs_adam_comparison.png # GD vs Adam results \
├── adam_vs_newton_comparison.png # Adam vs Newton results \
└── README.md # This file

🏃‍♂️ How to Run

1. Install dependencies:

bash
pip install - r requirement.txt

1. Run the complete comparison:

bash
python main.py
`

# Implementation here

"
📊 Output Files

The implementation generates:

1. gd_vs_adam_comparison.png - 4-panel comparison of GD vs Adam
2. adam_vs_newton_comparison.png - 4-panel comparison of Adam vs Newton
3. Console reports - Detailed performance analysis for both comparisons

🎓 Educational Value

This project demonstrates:

1. Optimizer Theory: Practical implementation of optimization algorithms
2. Convergence Behavior: How different optimizers navigate loss landscapes
3. Computational Trade-offs: Performance vs efficiency considerations
4. Regularization Effects: How L2 and Dropout improve generalization
5. Numerical Stability: Challenges with second-order methods

🔍 Key Insights

1. Adaptive methods (Adam) generally outperform fixed learning rate methods
2. Second-order methods are theoretically optimal but practically challenging
3. Mini-batch training provides good balance between stability and speed
4. Regularization is essential for good generalization performance
5. Computational cost must be considered alongside theoretical benefits

📄 License

This project is open source and available under the MIT License.
