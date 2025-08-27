import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer

def load_fashion_mnist(subset_size=None, test_size=0.2, random_state=42):
    """
    Load Fashion-MNIST dataset with optional subset
    
    Args:
        subset_size: Number of samples to use (None for full dataset)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed training and testing data
    """
    # Load Fashion-MNIST dataset
    print("Loading Fashion-MNIST dataset...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    
    # Convert to numpy arrays
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    
    # Use subset if specified
    if subset_size is not None and subset_size < len(X):
        indices = np.random.choice(len(X), subset_size, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"Using subset of {subset_size} samples")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encode the labels
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    
    return X_train, X_test, y_train, y_test