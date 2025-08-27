import numpy as np
from layers import DenseLayer

class NeuralNetwork:
    def __init__(self, layer_sizes, dropout_rates=None, l2_lambda=0.0):
        self.layers = []
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []
        
        if dropout_rates is None:
            dropout_rates = [0.0] * (len(layer_sizes) - 1)
        
        # Input layer
        self.layers.append(DenseLayer(layer_sizes[0], layer_sizes[1], 
                                     activation='relu', 
                                     dropout_rate=dropout_rates[0],
                                     l2_lambda=l2_lambda))
        
        # Hidden layers
        for i in range(1, len(layer_sizes) - 2):
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], 
                                         activation='relu', 
                                         dropout_rate=dropout_rates[i],
                                         l2_lambda=l2_lambda))
        
        # Output layer
        self.layers.append(DenseLayer(layer_sizes[-2], layer_sizes[-1], 
                                     activation='softmax', 
                                     dropout_rate=0.0,  # No dropout on output layer
                                     l2_lambda=l2_lambda))
    
    def forward(self, X, training=True):
        a = X
        for layer in self.layers:
            a = layer.forward(a, training=training)
        return a
    
    def backward(self, dA, learning_rate):
        d = dA
        for layer in reversed(self.layers):
            d = layer.backward(d, learning_rate)
    
    def compute_loss(self, y_true, y_pred, batch_size):
        # Categorical cross-entropy loss
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        # Add L2 regularization loss
        reg_loss = 0
        for layer in self.layers:
            reg_loss += layer.get_regularization_loss(batch_size)
        
        return loss + reg_loss
    
    def compute_accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == labels)