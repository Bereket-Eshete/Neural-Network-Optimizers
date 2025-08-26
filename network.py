# network.py
import numpy as np
# We will import the specific optimizers when we use them

class Network:
    """A neural network model that consists of a sequence of layers."""
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def set_loss(self, loss):
        """Set the loss function for the network."""
        self.loss = loss

    def set_optimizer(self, optimizer):
        """Set the optimizer for the network."""
        self.optimizer = optimizer

    def forward(self, x, is_training=True):
        """
        Perform a forward pass through all layers of the network.
        
        Args:
            x: Input data.
            is_training: Boolean flag for dropout.
            
        Returns:
            Output of the final layer.
        """
        # Pass data through each layer sequentially
        for layer in self.layers:
            x = layer.forward(x, is_training)
        return x

    def backward(self, dout):
        """
        Perform a backward pass through all layers of the network.
        
        Args:
            dout: Gradient from the loss function.
        """
        # Pass gradient backwards through each layer in reverse order
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update_parameters(self, learning_rate):
        """Update parameters in all layers using the optimizer."""
        for layer in self.layers:
            # Only layers with parameters (weights/biases) need updating
            if hasattr(layer, 'weights'):
                layer.update_parameters(self.optimizer, learning_rate)

    def get_accuracy(self, y_pred, y_true):
        """Calculate accuracy by comparing predicted and true labels."""
        # For one-hot encoded labels, find the index of the max value
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == labels)