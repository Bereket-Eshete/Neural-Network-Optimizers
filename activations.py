# activations.py
import numpy as np

class ReLU:
    """ReLU Activation Function"""
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        # Gradient is 1 where input > 0, else 0
        dinput = dout.copy()
        dinput[self.input <= 0] = 0
        return dinput

class Softmax:
    """Softmax Activation Function"""
    def forward(self, x):
        # Stabilize by subtracting max value
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dout):
        # The derivative will be handled in the CrossEntropy loss
        # for numerical stability. We just pass the gradient through here.
        return dout