import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.name = "Gradient Descent"
    
    def update(self, network, X_batch, y_batch):
        # Forward pass
        y_pred = network.forward(X_batch, training=True)
        
        # Compute loss gradient
        batch_size = X_batch.shape[0]
        dA = (y_pred - y_batch) / batch_size
        
        # Backward pass
        network.backward(dA, self.learning_rate)
        
        return network.compute_loss(y_batch, y_pred, batch_size)

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None
        self.name = "Adam"
    
    def initialize_moments(self, network):
        """Initialize moment estimates for all parameters in the network"""
        self.m = []
        self.v = []
        
        for layer in network.layers:
            self.m.append({
                'w': np.zeros_like(layer.weights),
                'b': np.zeros_like(layer.biases)
            })
            self.v.append({
                'w': np.zeros_like(layer.weights),
                'b': np.zeros_like(layer.biases)
            })
    
    def update(self, network, X_batch, y_batch):
        self.t += 1
        
        # Initialize moments if not done yet
        if self.m is None:
            self.initialize_moments(network)
        
        # Forward pass
        y_pred = network.forward(X_batch, training=True)
        
        # Compute loss gradient
        batch_size = X_batch.shape[0]
        dA = (y_pred - y_batch) / batch_size
        
        # Backward pass to compute gradients
        d = dA
        gradients = []
        
        # Compute gradients for each layer (without updating parameters)
        for i, layer in enumerate(reversed(network.layers)):
            # Apply dropout mask if needed
            if layer.dropout_rate > 0 and layer.dropout_mask is not None:
                d *= layer.dropout_mask
                d /= (1 - layer.dropout_rate)
            
            # Compute gradients
            if layer.activation == 'relu':
                dZ = d * (layer.output > 0).astype(float)
            elif layer.activation == 'softmax':
                dZ = d
            else:
                dZ = d
            
            dW = np.dot(layer.input.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            dX = np.dot(dZ, layer.weights.T)
            
            # Add L2 regularization gradient
            if layer.l2_lambda > 0:
                dW += layer.l2_lambda * layer.weights
            
            gradients.insert(0, {'dW': dW, 'db': db})
            d = dX
        
        # Update parameters using Adam
        for i, (layer, grad) in enumerate(zip(network.layers, gradients)):
            # Update moments for weights
            self.m[i]['w'] = self.beta1 * self.m[i]['w'] + (1 - self.beta1) * grad['dW']
            self.v[i]['w'] = self.beta2 * self.v[i]['w'] + (1 - self.beta2) * (grad['dW'] ** 2)
            
            # Update moments for biases
            self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * grad['db']
            self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * (grad['db'] ** 2)
            
            # Bias correction
            m_hat_w = self.m[i]['w'] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v[i]['w'] / (1 - self.beta2 ** self.t)
            
            m_hat_b = self.m[i]['b'] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v[i]['b'] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        
        return network.compute_loss(y_batch, y_pred, batch_size)