# activations.py
import numpy as np

class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.
    f(x) = max(0, x)
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.trainable = False  # Activation layers have no trainable parameters

    def forward(self, input):
        """
        Performs the forward pass: output = max(0, input)
        Args:
            input: Input data of any shape
        Returns:
            output: Activated output, same shape as input
        """
        self.input = input
        self.output = np.maximum(0, input)
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Performs the backward pass.
        The gradient is 1 where input > 0, else 0.
        Args:
            output_error: dE/dY of the next layer
            learning_rate: Not used in ReLU, but required by interface
        Returns:
            input_error: dE/dX = dE/dY * (1 if input > 0 else 0)
        """
        # Create mask where input > 0
        relu_mask = (self.input > 0).astype(float)
        # Gradient flows only where input was positive
        input_error = output_error * relu_mask
        return input_error

class Softmax:
    """
    Softmax activation function.
    Converts raw scores into probabilities that sum to 1.
    Should be used in the output layer for classification.
    """
    def __init__(self):
        self.input = None
        self.output = None
        self.trainable = False  # Activation layers have no trainable parameters

    def forward(self, input):
        """
        Performs the forward pass: output = exp(input) / sum(exp(input))
        Args:
            input: Input data of shape (batch_size, num_classes)
        Returns:
            output: Probability distribution, same shape as input
        """
        self.input = input
        # Numerical stability: subtract max from each row to avoid large exponents
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Performs the backward pass.
        The gradient of softmax is complex, but when combined with
        cross-entropy loss, the gradient calculation simplifies dramatically.
        However, we implement the general case for completeness.
        
        Args:
            output_error: dE/dY of the next layer (usually from loss function)
            learning_rate: Not used in Softmax, but required by interface
        Returns:
            input_error: dE/dX
        """
        # For the general case, we need to compute the Jacobian matrix
        # This is computationally expensive: O(n^2) where n is num_classes
        batch_size = self.output.shape[0]
        num_classes = self.output.shape[1]
        
        input_error = np.zeros_like(self.output)
        
        # For each sample in the batch
        for i in range(batch_size):
            # Create Jacobian matrix for this sample
            jacobian = np.diag(self.output[i]) - np.outer(self.output[i], self.output[i])
            # Multiply by the output error to get input error
            input_error[i] = np.dot(output_error[i], jacobian)
            
        return input_error

# Note: In practice, when Softmax is used with Categorical Cross-Entropy loss,
# the backward pass is simplified to (y_pred - y_true). This simplification
# is handled by the loss function's backward method, so the Softmax backward
# might not actually be called in our final implementation.