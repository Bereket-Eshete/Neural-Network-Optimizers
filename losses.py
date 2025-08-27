# losses.py
import numpy as np

class CategoricalCrossentropy:
    """
    Categorical Cross-Entropy loss function.
    Used for multi-class classification (e.g., with Softmax output layer).
    """
    def __init__(self):
        self.y_pred = None
        self.y_true = None
        self.eps = 1e-12  # Small constant to avoid log(0)

    def forward(self, y_pred, y_true):
        """
        Computes the categorical cross-entropy loss.
        Args:
            y_pred: Model predictions (post-softmax), shape (batch_size, num_classes)
            y_true: Ground truth labels in one-hot encoding, shape (batch_size, num_classes)
        Returns:
            loss: The average cross-entropy loss over the batch.
        """
        # Store for backward pass
        self.y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)  # Clip to avoid log(0)
        self.y_true = y_true
        
        # Calculate cross-entropy loss: -sum(y_true * log(y_pred)) / batch_size
        batch_losses = -np.sum(y_true * np.log(self.y_pred), axis=1)
        loss = np.mean(batch_losses)
        return loss

    def backward(self):
        """
        Computes the gradient of the loss with respect to the predictions (pre-softmax).
        The gradient is: (y_pred - y_true) / batch_size
        This form assumes that the loss is followed by a Softmax activation.
        Returns:
            dinputs: Gradient of the loss w.r.t. the pre-softmax inputs, shape (batch_size, num_classes)
        """
        batch_size = self.y_true.shape[0]
        # Gradient of cross-entropy loss w.r.t. softmax output is (y_pred - y_true)
        dinputs = (self.y_pred - self.y_true) / batch_size
        return dinputs

    def calculate_regularization_loss(self, layers):
        """
        Calculates the L2 regularization loss from all trainable layers.
        Args:
            layers: List of all layers in the network.
        Returns:
            regularization_loss: The total L2 regularization loss.
        """
        regularization_loss = 0
        
        for layer in layers:
            # Check if layer is trainable and has L2 regularization
            if hasattr(layer, 'weight_regularizer_l2') and layer.weight_regularizer_l2 > 0:
                # L2 regularization: lambda * sum(weights^2)
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
                
        return regularization_loss

    def total_loss(self, y_pred, y_true, layers):
        """
        Computes the total loss: data loss + regularization loss.
        Args:
            y_pred: Model predictions (post-softmax)
            y_true: Ground truth labels
            layers: List of all layers in the network
        Returns:
            total_loss: data_loss + regularization_loss
        """
        data_loss = self.forward(y_pred, y_true)
        regularization_loss = self.calculate_regularization_loss(layers)
        total_loss = data_loss + regularization_loss
        return total_loss