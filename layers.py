import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu', dropout_rate=0.0, l2_lambda=0.0):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.input = None
        self.output = None
        self.dropout_mask = None
    
    def forward(self, X, training=True):
        self.input = X
        z = np.dot(X, self.weights) + self.biases
        
        if self.activation == 'relu':
            a = np.maximum(0, z)
        elif self.activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            a = z  # Linear activation
        
        # Apply dropout if training
        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
            a *= self.dropout_mask
            a /= (1 - self.dropout_rate)  # Scale during training
        
        self.output = a
        return self.output
    
    def backward(self, dA, learning_rate):
        # If dropout was applied, mask the gradients
        if self.dropout_rate > 0 and self.dropout_mask is not None:
            dA *= self.dropout_mask
            dA /= (1 - self.dropout_rate)
        
        if self.activation == 'relu':
            dZ = dA * (self.output > 0).astype(float)
        elif self.activation == 'softmax':
            dZ = dA  # For softmax, the derivative is handled in the loss function
        else:
            dZ = dA  # Linear activation
        
        # Calculate gradients
        dW = np.dot(self.input.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.weights.T)
        
        # Add L2 regularization gradient
        if self.l2_lambda > 0:
            dW += self.l2_lambda * self.weights
        
        # Update parameters
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        
        return dX
    
    def get_regularization_loss(self, batch_size):
        if self.l2_lambda > 0:
            return 0.5 * self.l2_lambda * np.sum(self.weights ** 2) / batch_size
        return 0