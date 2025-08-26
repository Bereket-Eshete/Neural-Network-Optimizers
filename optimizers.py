# optimizers.py
import numpy as np

class GradientDescent:
    """Standard Gradient Descent optimizer."""
    def __init__(self):
        self.name = "Gradient Descent"

    def update(self, weights, biases, dweights, dbiases, learning_rate):
        """
        Performs a simple gradient descent update.
        
        Args:
            weights: Current weights of the layer.
            biases: Current biases of the layer.
            dweights: Gradient of the loss w.r.t. weights.
            dbiases: Gradient of the loss w.r.t. biases.
            learning_rate: The learning rate (eta).
            
        Returns:
            updated_weights, updated_biases
        """
        weights_updated = weights - learning_rate * dweights
        biases_updated = biases - learning_rate * dbiases
        return weights_updated, biases_updated

class Adam:
    """Adam (Adaptive Moment Estimation) optimizer."""
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.name = "Adam"
        self.beta1 = beta1  # Decay rate for first moment (mean)
        self.beta2 = beta2  # Decay rate for second moment (uncentered variance)
        self.epsilon = epsilon # Small number to prevent division by zero
        self.t = 0 # Time step counter
        
        # Initialize moment estimates for weights and biases for each layer
        # These will be set in the `update` method when we see the parameters for the first time
        self.m_weights = None
        self.v_weights = None
        self.m_biases = None
        self.v_biases = None

    def update(self, weights, biases, dweights, dbiases, learning_rate):
        """
        Performs an Adam update on the parameters.
        """
        # Initialize moment estimates if this is the first call for these parameters
        if self.m_weights is None:
            self.m_weights = np.zeros_like(weights)
            self.v_weights = np.zeros_like(weights)
            self.m_biases = np.zeros_like(biases)
            self.v_biases = np.zeros_like(biases)
            self.t = 0
        
        # Increase the time step
        self.t += 1
        
        # Update biased first moment estimate (WEIGHTS)
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * dweights
        # Update biased second moment estimate (WEIGHTS)
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (dweights ** 2)
        
        # Update biased first moment estimate (BIASES)
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * dbiases
        # Update biased second moment estimate (BIASES)
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (dbiases ** 2)
        
        # Compute bias-corrected first moment estimate (WEIGHTS)
        m_hat_weights = self.m_weights / (1 - self.beta1 ** self.t)
        # Compute bias-corrected second moment estimate (WEIGHTS)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** self.t)
        
        # Compute bias-corrected first moment estimate (BIASES)
        m_hat_biases = self.m_biases / (1 - self.beta1 ** self.t)
        # Compute bias-corrected second moment estimate (BIASES)
        v_hat_biases = self.v_biases / (1 - self.beta2 ** self.t)
        
        # Update parameters (WEIGHTS)
        weights_updated = weights - learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        # Update parameters (BIASES)
        biases_updated = biases - learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)
        
        return weights_updated, biases_updated