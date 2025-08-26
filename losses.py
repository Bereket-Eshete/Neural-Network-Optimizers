# losses.py
import numpy as np

class CrossEntropyLoss:
    """Cross Entropy Loss (assumes Softmax is used in the output layer)"""
    def forward(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate loss for each sample, then average
        sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return np.mean(sample_losses)
    
    def backward(self):
        # The gradient of the combined Softmax + CrossEntropy loss
        # is (y_pred - y_true)
        return (self.y_pred - self.y_true) / self.y_true.shape[0] # Normalize by batch size