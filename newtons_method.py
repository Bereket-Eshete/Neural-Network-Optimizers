import numpy as np
import time

class EfficientNewtonsMethod:
    def __init__(self, learning_rate=0.1, epsilon=1e-6, h=1e-4):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.h = h  # Step size for numerical differentiation
        self.name = "Newton's Method"
        self.loss_history = []
    
    def compute_gradient_approximation(self, network, X_batch, y_batch):
        """Compute gradient approximation efficiently"""
        batch_size = X_batch.shape[0]
        original_params = self._get_parameters(network)
        gradient = []
        
        # Forward pass to get current loss
        y_pred = network.forward(X_batch, training=False)
        current_loss = network.compute_loss(y_batch, y_pred, batch_size)
        
        for i, layer in enumerate(network.layers):
            grad_w = np.zeros_like(layer.weights)
            grad_b = np.zeros_like(layer.biases)
            
            # Compute gradient for weights (sample first 100)
            flat_weights = layer.weights.flatten()
            for idx in range(min(100, len(flat_weights))):
                row = idx // layer.weights.shape[1]
                col = idx % layer.weights.shape[1]
                
                layer.weights[row, col] += self.h
                y_pred_plus = network.forward(X_batch, training=False)
                loss_plus = network.compute_loss(y_batch, y_pred_plus, batch_size)
                
                # Reset weight
                layer.weights[row, col] -= self.h
                
                grad_w[row, col] = (loss_plus - current_loss) / self.h
            
            # Compute gradient for biases (sample first 10)
            for col in range(min(10, layer.biases.shape[1])):
                layer.biases[0, col] += self.h
                y_pred_plus = network.forward(X_batch, training=False)
                loss_plus = network.compute_loss(y_batch, y_pred_plus, batch_size)
                
                layer.biases[0, col] -= self.h
                grad_b[0, col] = (loss_plus - current_loss) / self.h
            
            gradient.append({'w': grad_w, 'b': grad_b})
        
        self._set_parameters(network, original_params)
        return gradient
    
    def compute_diagonal_hessian_approximation(self, network, X_batch, y_batch):
        """Compute diagonal Hessian approximation efficiently"""
        batch_size = X_batch.shape[0]
        original_params = self._get_parameters(network)
        hessian_diag = []
        
        # Forward pass to get current loss
        y_pred = network.forward(X_batch, training=False)
        current_loss = network.compute_loss(y_batch, y_pred, batch_size)
        
        for i, layer in enumerate(network.layers):
            hessian_w = np.zeros_like(layer.weights)
            hessian_b = np.zeros_like(layer.biases)
            
            # Compute diagonal Hessian for weights (approximate)
            flat_weights = layer.weights.flatten()
            for idx in range(min(100, len(flat_weights))):  # Only sample 100 weights for efficiency
                row = idx // layer.weights.shape[1]
                col = idx % layer.weights.shape[1]
                
                # Perturb weight and compute loss change
                layer.weights[row, col] += self.h
                y_pred_plus = network.forward(X_batch, training=False)
                loss_plus = network.compute_loss(y_batch, y_pred_plus, batch_size)
                
                layer.weights[row, col] -= 2 * self.h
                y_pred_minus = network.forward(X_batch, training=False)
                loss_minus = network.compute_loss(y_batch, y_pred_minus, batch_size)
                
                # Reset weight
                layer.weights[row, col] += self.h
                
                # Diagonal Hessian approximation
                hessian_w[row, col] = (loss_plus - 2 * current_loss + loss_minus) / (self.h ** 2)
            
            # Compute diagonal Hessian for biases (sample first few)
            for col in range(min(10, layer.biases.shape[1])):
                layer.biases[0, col] += self.h
                y_pred_plus = network.forward(X_batch, training=False)
                loss_plus = network.compute_loss(y_batch, y_pred_plus, batch_size)
                
                layer.biases[0, col] -= 2 * self.h
                y_pred_minus = network.forward(X_batch, training=False)
                loss_minus = network.compute_loss(y_batch, y_pred_minus, batch_size)
                
                layer.biases[0, col] += self.h
                hessian_b[0, col] = (loss_plus - 2 * current_loss + loss_minus) / (self.h ** 2)
            
            hessian_diag.append({'w': hessian_w, 'b': hessian_b})
        
        self._set_parameters(network, original_params)
        return hessian_diag
    
    def _get_parameters(self, network):
        """Get all parameters from network"""
        params = []
        for layer in network.layers:
            params.append({
                'weights': layer.weights.copy(),
                'biases': layer.biases.copy()
            })
        return params
    
    def _set_parameters(self, network, params):
        """Set all parameters in network"""
        for i, layer in enumerate(network.layers):
            layer.weights = params[i]['weights']
            layer.biases = params[i]['biases']
    
    def update(self, network, X_batch, y_batch):
        """Efficient Newton's method update using diagonal Hessian approximation"""
        print("Computing Newton update (this may take a moment)...")
        start_time = time.time()
        
        # Compute gradient and Hessian approximations
        gradient = self.compute_gradient_approximation(network, X_batch, y_batch)
        hessian_diag = self.compute_diagonal_hessian_approximation(network, X_batch, y_batch)
        
        # Update each layer with careful step size
        for i, layer in enumerate(network.layers):
            # Regularize Hessian to avoid division by zero and extreme values
            hessian_w = np.abs(hessian_diag[i]['w']) + self.epsilon
            hessian_b = np.abs(hessian_diag[i]['b']) + self.epsilon
            
            # Clip extreme Hessian values
            hessian_w = np.clip(hessian_w, 1e-6, 1e6)
            hessian_b = np.clip(hessian_b, 1e-6, 1e6)
            
            # Update weights and biases using Newton's step with careful scaling
            layer.weights -= self.learning_rate * gradient[i]['w'] / hessian_w
            layer.biases -= self.learning_rate * gradient[i]['b'] / hessian_b
        
        # Compute final loss
        y_pred = network.forward(X_batch, training=False)
        loss = network.compute_loss(y_batch, y_pred, X_batch.shape[0])
        self.loss_history.append(loss)
        
        end_time = time.time()
        print(f"Newton update completed in {end_time - start_time:.2f} seconds")
        
        return loss