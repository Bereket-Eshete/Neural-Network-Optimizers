import numpy as np

def train_model(network, optimizer, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train a neural network with the specified optimizer
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    # Check if this is Newton's method
    is_newton = hasattr(optimizer, 'name') and "Newton" in optimizer.name
    
    if is_newton:
        n_batches = 1  # Newton's method uses only one batch per epoch
    else:
        n_batches = int(np.ceil(X_train.shape[0] / batch_size))
    
    for epoch in range(epochs):
        if is_newton:
            # For Newton's Method, use a single batch per epoch due to computational cost
            X_batch = X_train[:batch_size]
            y_batch = y_train[:batch_size]
            
            print(f"Epoch {epoch}: Computing Newton update...")
            batch_loss = optimizer.update(network, X_batch, y_batch)
            epoch_loss = batch_loss
            
        else:
            # Standard mini-batch training for other optimizers
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, X_train.shape[0])
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                batch_loss = optimizer.update(network, X_batch, y_batch)
                epoch_loss += batch_loss
            
            epoch_loss /= n_batches
        
        history['train_loss'].append(epoch_loss)
        
        # Validation
        y_val_pred = network.forward(X_val, training=False)
        val_loss = network.compute_loss(y_val, y_val_pred, X_val.shape[0])
        val_accuracy = network.compute_accuracy(y_val, y_val_pred)
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Training accuracy
        y_train_pred = network.forward(X_train, training=False)
        train_accuracy = network.compute_accuracy(y_train, y_train_pred)
        history['train_accuracy'].append(train_accuracy)
        
        if epoch % 1 == 0:  # Print every epoch
            print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Val Loss = {val_loss:.4f}, "
                  f"Train Acc = {train_accuracy:.4f}, Val Acc = {val_accuracy:.4f}")
    
    return history