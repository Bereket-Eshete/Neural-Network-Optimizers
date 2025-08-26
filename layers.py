# layers.py
import numpy as np

class Layer:
    """A fully-connected neural network layer."""
    def __init__(self, n_input, n_neurons, activation=None, l2_lambda=0.0, dropout_rate=0.0):
        # Initialize weights and biases
        # He initialization is good for ReLU
        self.weights = np.random.randn(n_input, n_neurons) * np.sqrt(2. / n_input)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        
        # Cache for backward pass
        self.input = None
        self.output = None
        self.dropout_mask = None

    def forward(self, x, is_training=True):
        self.input = x
        # Linear transformation
        z = np.dot(x, self.weights) + self.biases
        
        # Apply activation function (if any)
        if self.activation is not None:
            a = self.activation.forward(z)
        else:
            a = z
        
        # Apply dropout during training
        if is_training and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape) / (1 - self.dropout_rate)
            a = a * self.dropout_mask
            
        self.output = a
        return self.output

    def backward(self, dout):
        # 1. Backpropagate through dropout
        if self.dropout_rate > 0.0:
            dout = dout * self.dropout_mask
            
        # 2. Backpropagate through activation function
        if self.activation is not None:
            dout = self.activation.backward(dout)
            
        # 3. Calculate gradients for weights and biases
        batch_size = self.input.shape[0]
        dweights = np.dot(self.input.T, dout)
        dbiases = np.sum(dout, axis=0, keepdims=True)
        
        # Add L2 regularization gradient
        if self.l2_lambda > 0:
            dweights += self.l2_lambda * self.weights
            
        self.dweights = dweights
        self.dbiases = dbiases
        
        # 4. Calculate gradient for the input (to pass to previous layer)
        dinput = np.dot(dout, self.weights.T)
        
        return dinput

    def update_parameters(self, optimizer, learning_rate):
        """Update weights and biases using the provided optimizer."""
        self.weights, self.biases = optimizer.update(self.weights, self.biases, self.dweights, self.dbiases, learning_rate)